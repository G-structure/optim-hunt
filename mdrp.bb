#!/usr/bin/env bb

;;; ===================================================================
;;; === RunPod Markdown Executor ===
;;; ===================================================================
;;; This script provides a TUI to execute code blocks in markdown files
;;; using RunPod. It allows you to:
;;; 1. Select markdown files in the current directory
;;; 2. Use existing RunPod pods or create new ones
;;; 3. Execute the code blocks remotely using site-gen.bb
;;; 4. Wait for a configurable idle time before stopping the pod
;;; 5. Automatically stop the pod when execution is complete
;;;
;;; The script integrates with the existing site-gen.bb functionality
;;; but adds RunPod management capabilities.

(ns runpod-executor
  "Provides a terminal UI for executing markdown code blocks on RunPod instances."
  (:require [babashka.process :as process]
            [clojure.string :as str]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [babashka.fs :as fs]))

;;; ===================================================================
;;; === Configuration ===
;;; ===================================================================

(def default-config
  "Default configuration settings for the executor."
  {:pod-idle-timeout 300    ; Seconds to wait before stopping pod after execution
   :gpu-type "NVIDIA GeForce RTX 3070"  ; Default GPU type
   :gpu-count 1             ; Default number of GPUs
   :container-disk 40       ; Container disk size in GB
   :volume-size 100         ; Volume size in GB
   :image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"  ; Default Docker image
   :pod-name "RunPod Markdown Executor"  ; Default pod name
   :ssh-key "~/.ssh/id_ed25519" ; Default SSH key path
   :site-gen-script "./site-gen.bb"})  ; Path to site-gen.bb script

;; Runtime configuration (from config + command line args)
(def config (atom default-config))

;;; ===================================================================
;;; === TUI Components and Utilities ===
;;; ===================================================================

(defn clear-screen
  "Clears the terminal screen."
  []
  (print (str (char 27) "[2J" (char 27) "[H"))
  (flush))

(defn prompt
  "Displays a prompt and reads a line of input."
  [message]
  (print message)
  (flush)
  (read-line))

(defn prompt-with-default
  "Displays a prompt with default value and reads input."
  [message default]
  (let [input (prompt (format "%s [%s]: " message default))]
    (if (str/blank? input) default input)))

(defn prompt-number
  "Prompts for a number with default value."
  [message default]
  (let [input (prompt (format "%s [%s]: " message default))]
    (if (str/blank? input)
      default
      (try
        (Integer/parseInt input)
        (catch Exception _
          (println "Invalid number, using default.")
          default)))))

(defn prompt-yn
  "Prompts for a yes/no response."
  [message default]
  (let [default-str (if default "Y/n" "y/N")
        input (prompt (format "%s [%s]: " message default-str))]
    (if (str/blank? input)
      default
      (contains? #{"y" "Y" "yes" "Yes" "YES"} input))))

(defn display-spinner
  "Displays a spinning animation while an operation is in progress."
  [message operation-fn]
  (let [spinner-chars "|/-\\"
        running (atom true)
        spinner-thread (future
                         (try
                           (let [out *out*]
                             (loop [i 0]
                               (when @running
                                 (binding [*out* out]
                                   (print (str "\r" message " " (nth spinner-chars (mod i (count spinner-chars)))))
                                   (flush))
                                 (Thread/sleep 100)
                                 (recur (inc i)))))
                           (catch Exception _ nil)))]
    (try
      (let [result (operation-fn)]
        (reset! running false)
        (deref spinner-thread 100 nil)
        (println "\r" message " Done!      ")
        result)
      (catch Exception e
        (reset! running false)
        (deref spinner-thread 100 nil)
        (println "\r" message " Failed!    ")
        (throw e)))))

(defn format-uptime
  "Formats uptime seconds into a human-readable string."
  [seconds]
  (if (nil? seconds)
    "N/A"
    (let [days (quot seconds 86400)
          hours (rem (quot seconds 3600) 24)
          minutes (rem (quot seconds 60) 60)
          seconds (rem seconds 60)]
      (cond
        (pos? days) (format "%dd %dh %dm" days hours minutes)
        (pos? hours) (format "%dh %dm" hours minutes)
        :else (format "%dm %ds" minutes seconds)))))

(defn find-md-files
  "Finds all markdown files in the current directory."
  []
  (let [files (fs/glob "." "*.md")]
    (mapv #(fs/file-name %) files)))

(defn display-md-files
  "Displays markdown files in a numbered list."
  [files]
  (println "\n=== Markdown Files ===")
  (if (empty? files)
    (println "No markdown files found in the current directory.")
    (doseq [[idx file] (map-indexed vector files)]
      (println (format "[%d] %s" (inc idx) file)))))

(defn default-output-filename
  "Generates a default output filename from a markdown filename."
  [md-file]
  (-> md-file
      (str/replace #"\.md$" ".html")))

;;; ===================================================================
;;; === RunPod API Interaction ===
;;; ===================================================================

(defn make-graphql-request
  "Makes a GraphQL request to the RunPod API.

   Args:
   query - GraphQL query/mutation string
   api-key - RunPod API key
   variables - Optional map of variables for the query

   Returns:
   Parsed JSON response or nil on failure"
  [query api-key & [variables]]
  (try
    (let [request-body (json/generate-string
                        {:query query
                         :variables (or variables {})})
          response (process/process ["curl"
                                     "--silent"
                                     "--request" "POST"
                                     "--header" "Content-Type: application/json"
                                     "--header" (str "Authorization: Bearer " api-key)
                                     "--url" "https://api.runpod.io/graphql"
                                     "--data" request-body]
                                    {:out :string})]
      (-> response
          process/check
          :out
          (json/parse-string true)))
    (catch Exception e
      (println "API request failed:" (ex-message e))
      nil)))

(defn fetch-gpu-types
  "Retrieves available GPU types from RunPod.

   Args:
   api-key - RunPod API key

   Returns:
   Vector of GPU types with id and display name"
  [api-key]
  (println "Fetching GPU types...")
  (let [query "query GpuTypes { gpuTypes { id displayName memoryInGb secureCloud communityCloud } }"
        response (make-graphql-request query api-key)]
    (if-let [gpu-types (get-in response [:data :gpuTypes])]
      (do
        (println "Found" (count gpu-types) "GPU types")
        gpu-types)
      (do
        (println "Failed to fetch GPU types")
        []))))

(defn get-user-pods
  "Retrieves all pods belonging to the user.

   Args:
   api-key - RunPod API key

   Returns:
   List of pods or empty list on failure"
  [api-key]
  (println "Fetching your pods...")
  (let [query "query Pods {
               myself {
                 pods {
                   id
                   name
                   desiredStatus
                   machineId
                   runtime {
                     uptimeInSeconds
                     ports {
                       ip
                       isIpPublic
                       privatePort
                       publicPort
                       type
                     }
                   }
                 }
               }
             }"
        response (make-graphql-request query api-key)]
    (if-let [pods (get-in response [:data :myself :pods])]
      (do
        (println "Found" (count pods) "pods")
        pods)
      (do
        (println "Failed to fetch pods")
        []))))

(defn create-pod
  "Creates a new RunPod pod.

   Args:
   api-key - RunPod API key
   config - Map with pod configuration

   Returns:
   Created pod data or nil on failure"
  [api-key pod-config]
  (println "\n=== Creating RunPod Pod ===")
  (let [{:keys [gpu-type gpu-count pod-name image container-disk volume-size]} pod-config
        mutation "mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
   podFindAndDeployOnDemand(input: $input) {
   id name imageName desiredStatus machineId
   }
   }"
        variables {:input {:gpuTypeId gpu-type
                           :gpuCount gpu-count
                           :name pod-name
                           :imageName image
                           :containerDiskInGb container-disk
                           :volumeInGb volume-size
                           :cloudType "ALL"
                           :volumeMountPath "/workspace"
                           :ports "22/tcp,8888/http"}}]  ; Explicitly prioritize SSH port

    (println "Creating pod with:")
    (println (str "  GPU Type: " gpu-type))
    (println (str "  GPU Count: " gpu-count))
    (println (str "  Name: " pod-name))
    (println (str "  Image: " image))
    (println (str "  Ports: 22/tcp,8888/http"))

    (let [response (display-spinner "Creating pod"
                                    #(make-graphql-request mutation api-key variables))]
      (if-let [pod (get-in response [:data :podFindAndDeployOnDemand])]
        (do
          (println "Pod created successfully:")
          (println (str "  Pod ID: " (:id pod)))
          (println (str "  Status: " (:desiredStatus pod)))
          pod)
        (do
          (println "Failed to create pod:")
          (when-let [errors (:errors response)]
            (println (str "  Error: " (get-in errors [0 :message] "Unknown error"))))
          nil)))))

(defn get-pod-details
  "Gets detailed pod information including runtime and ports.

   Args:
   api-key - RunPod API key
   pod-id - ID of the pod to query

   Returns:
   Detailed pod data or nil on failure"
  [api-key pod-id]
  (let [query "query PodDetails($input: PodFilter!) {
   pod(input: $input) {
   id
   desiredStatus
   runtime {
   uptimeInSeconds
   ports {
   publicPort
   privatePort
   type
   ip
   isIpPublic
   }
   }
   }
   }"
        response (make-graphql-request query api-key {:input {:podId pod-id}})]
    (get-in response [:data :pod])))

(defn start-pod
  "Starts an existing pod.

   Args:
   api-key - RunPod API key
   pod-id - ID of the pod to start

   Returns:
   Updated pod data or nil on failure"
  [api-key pod-id]
  (println "\n=== Starting Pod ===")
  (println "Starting pod" pod-id)

  (let [mutation "mutation StartPod($input: PodResumeInput!) {
                    podResume(input: $input) {
                      id
                      desiredStatus
                    }
                  }"
        variables {:input {:podId pod-id}}
        response (make-graphql-request mutation api-key variables)]

    (if-let [pod (get-in response [:data :podResume])]
      (do
        (println "Pod started successfully:")
        (println (str "  Status: " (:desiredStatus pod)))
        pod)
      (do
        (println "Failed to start pod:")
        (when-let [errors (:errors response)]
          (println (str "  Error: " (get-in errors [0 :message] "Unknown error"))))
        nil))))

(defn wait-for-pod-running
  "Waits for a pod to reach the RUNNING state.

   Args:
   api-key - RunPod API key
   pod-id - ID of the pod to check
   timeout - Timeout in seconds (default: 600)

   Returns:
   Pod data when running, or nil on timeout"
  [api-key pod-id & {:keys [timeout] :or {timeout 600}}]
  (println "\n=== Waiting for Pod to Start ===")
  (println "Pod ID:" pod-id)
  (println "Timeout:" timeout "seconds")

  (let [start-time (System/currentTimeMillis)]
    (loop []
      (let [elapsed (int (/ (- (System/currentTimeMillis) start-time) 1000))
            _ (print (format "\rWaiting for pod... %d seconds elapsed" elapsed))
            _ (flush)
            pod (get-pod-details api-key pod-id)
            status (:desiredStatus pod)]

        (cond
          (> elapsed timeout)
          (do (println "\nTimed out waiting for pod to start")
              nil)

          (= status "RUNNING")
          (do (println "\nPod is now running!")
              pod)

          :else
          (do (Thread/sleep 5000)
              (recur)))))))

(defn wait-for-ssh-port
  "Waits for the SSH port to become available.

   Args:
   api-key - RunPod API key
   pod-id - ID of the pod to check
   timeout - Timeout in seconds (default: 300)

   Returns:
   Pod data with SSH port available, or nil on timeout"
  [api-key pod-id & {:keys [timeout] :or {timeout 300}}]
  (println "\n=== Waiting for SSH Port ===")
  (println "Pod ID:" pod-id)
  (println "Timeout:" timeout "seconds")

  (let [start-time (System/currentTimeMillis)]
    (loop []
      (let [elapsed (int (/ (- (System/currentTimeMillis) start-time) 1000))
            _ (print (format "\rWaiting for SSH port... %d seconds elapsed" elapsed))
            _ (flush)
            pod (get-pod-details api-key pod-id)
            ports (get-in pod [:runtime :ports])]

        ;; Debug port information
        (when (and (zero? (mod elapsed 30)) (seq ports))
          (println "\nAvailable ports:")
          (doseq [port ports]
            (println (format "  - Port %d -> %d (%s) on %s"
                             (:privatePort port)
                             (:publicPort port)
                             (:type port)
                             (:ip port)))))

        (cond
          (> elapsed timeout)
          (do (println "\nTimed out waiting for SSH port")
              nil)

          (some #(and (= (:privatePort %) 22) (= (:type %) "tcp") (:publicPort %)) ports)
          (do (println "\nSSH port is now available!")
              pod)

          :else
          (do (Thread/sleep 5000)
              (recur)))))))

(defn get-ssh-port
  "Finds the SSH port configuration for a pod.

   Args:
   pod - Pod data from API

   Returns:
   SSH port configuration or nil"
  [pod]
  (let [ports (get-in pod [:runtime :ports])]
    (when (seq ports)
      (println "\n=== Available Ports ===")
      (doseq [port ports]
        (println (format "Port %d -> %d (%s) on %s (Public: %s)"
                         (:privatePort port)
                         (:publicPort port)
                         (:type port)
                         (:ip port)
                         (:isIpPublic port))))

      ;; Find SSH port
      (let [ssh-port (first (filter #(and (= (:privatePort %) 22)) ports))]
        (if ssh-port
          (do
            (println "Found SSH port:" (:publicPort ssh-port))
            ssh-port)
          (println "No SSH port found"))))))

(defn get-pod-ssh-details
  "Extracts SSH connection details from pod data.

   Args:
   pod - Pod data from API

   Returns:
   Map with SSH connection details or nil"
  [pod]
  (println "\n=== Getting SSH Details ===")
  (when-let [ssh-port (get-ssh-port pod)]
    (let [ssh-details {:host (:ip ssh-port)
                       :public-host (when (:isIpPublic ssh-port) (:ip ssh-port))
                       :port (:publicPort ssh-port)
                       :private-port (:privatePort ssh-port)}]
      (println "SSH connection details:")
      (println (format "  Host: %s" (:host ssh-details)))
      (println (format "  Port: %d" (:port ssh-details)))
      (println (format "  Public: %s" (boolean (:public-host ssh-details))))
      ssh-details)))

(defn stop-pod
  "Stops a running pod.

   Args:
   api-key - RunPod API key
   pod-id - ID of the pod to stop

   Returns:
   true on success, false on failure"
  [api-key pod-id]
  (println "\n=== Stopping Pod ===")
  (println "Stopping pod" pod-id)

  (let [mutation "mutation StopPod($input: PodStopInput!) {
  podStop(input: $input) {
  id desiredStatus
  }
  }"
        response (make-graphql-request mutation api-key {:input {:podId pod-id}})]

    (if-let [errors (:errors response)]
      (do
        (println "Failed to stop pod:" (get-in errors [0 :message] "Unknown error"))
        false)
      (do
        (println "Pod stopped successfully")
        true))))

;;; ===================================================================
;;; === Site-Gen Integration ===
;;; ===================================================================

(defn prepare-site-gen-command
  "Prepares the site-gen.bb command for remote execution.

   Args:
   md-file - Markdown file to process
   output-file - Output HTML file to generate
   ssh-config - SSH connection details

   Returns:
   Command string array for process/exec"
  [md-file output-file ssh-config]
  (let [site-gen-script (:site-gen-script @config)
        host (or (:public-host ssh-config) (:host ssh-config))
        port (:port ssh-config)
        ssh-key (:ssh-key @config)]

    [site-gen-script
     md-file
     "-o" output-file
     "-sh" host
     "-sp" (str port)
     "-su" "root"
     "-sk" (fs/expand-home ssh-key)
     "-c"           ; Compute blocks
     "--recompute"]))  ; Always recompute code blocks

(defn execute-site-gen
  "Executes site-gen.bb with the given markdown file on the remote pod.

   Args:
   md-file - Markdown file to process
   output-file - Output HTML file to generate
   ssh-config - SSH connection details

   Returns:
   Map with execution details"
  [md-file output-file ssh-config]
  (println "\n=== Executing Code Blocks ===")
  (println "Processing file:" md-file)
  (println "Output file:" output-file)

  (try
    (let [cmd (prepare-site-gen-command md-file output-file ssh-config)
          _ (println "Command:" (str/join " " cmd))
          result (display-spinner (str "Processing " md-file)
                                  #(try
                                     (let [proc (process/process cmd {:out :string :err :string})]
                                       (process/check proc)
                                       {:exit 0
                                        :out (:out proc)
                                        :err (:err proc)})
                                     (catch Exception e
                                       {:exit 1
                                        :err (str "Error executing site-gen: " (.getMessage e))})))]

      (if (zero? (:exit result))
        (do
          (println "Code blocks executed successfully")
          (println "Output written to:" output-file)
          {:success true
           :output (:out result)})
        (do
          (println "Error executing code blocks:")
          (println (:err result))
          {:success false
           :error (:err result)})))
    (catch Exception e
      (println "Fatal error executing site-gen:" (.getMessage e))
      {:success false
       :error (.getMessage e)})))

;;; ===================================================================
;;; === Main Workflow Functions ===
;;; ===================================================================

(defn select-markdown-file
  "Prompts the user to select a markdown file.

   Returns:
   Selected markdown filename or nil"
  []
  (clear-screen)
  (println "=== RunPod Markdown Executor ===\n")
  (let [md-files (find-md-files)]
    (display-md-files md-files)
    (println "\n[0] Exit")

    (let [selection (prompt-number "\nSelect a file:" 0)]
      (if (= selection 0)
        nil
        (when (<= 1 selection (count md-files))
          (nth md-files (dec selection)))))))

(defn display-pod-list
  "Displays a list of pods with their status.

   Args:
   pods - List of pods from the API

   Returns:
   nil"
  [pods]
  (println "\n=== Available Pods ===")
  (if (empty? pods)
    (println "No pods found.")
    (do
      (println "  [0] Create new pod")
      (doseq [[idx pod] (map-indexed vector pods)]
        (let [status (:desiredStatus pod)
              uptime (get-in pod [:runtime :uptimeInSeconds])
              uptime-str (if uptime (format " (Up: %s)" (format-uptime uptime)) "")]
          (println (format "  [%d] %s - %s%s"
                           (inc idx)
                           (:name pod)
                           status
                           uptime-str)))))))

(defn select-pod
  "Prompts the user to select a pod from their account.

   Args:
   api-key - RunPod API key
   pods - List of pods (optional, will fetch if not provided)

   Returns:
   Selected pod or :create-new"
  [api-key & [provided-pods]]
  (let [pods (or provided-pods (get-user-pods api-key))]
    (display-pod-list pods)

    (let [selection (prompt-number "\nSelect a pod:" 0)]
      (cond
        (= selection 0) :create-new
        (<= 1 selection (count pods)) (nth pods (dec selection))
        :else nil))))

(defn configure-pod
  "Prompts the user for pod configuration options.

   Args:
   api-key - RunPod API key

   Returns:
   Map with pod configuration"
  [api-key]
  (clear-screen)
  (println "=== Configure RunPod ===\n")

  (let [gpu-types (fetch-gpu-types api-key)]
    (if (empty? gpu-types)
  ;; If we couldn't fetch GPU types, use hardcoded defaults
      (let [default-gpu-types [{:id "NVIDIA GeForce RTX 3070" :displayName "RTX 3070" :memoryInGb 8}
                               {:id "NVIDIA GeForce RTX 3080" :displayName "RTX 3080" :memoryInGb 10}
                               {:id "NVIDIA GeForce RTX 3090" :displayName "RTX 3090" :memoryInGb 24}
                               {:id "NVIDIA RTX A4000" :displayName "RTX A4000" :memoryInGb 16}
                               {:id "NVIDIA RTX A5000" :displayName "RTX A5000" :memoryInGb 24}
                               {:id "NVIDIA RTX A6000" :displayName "RTX A6000" :memoryInGb 48}]]

        ;; Display available GPU types
        (println "\nAvailable GPU Types (fallback list):")
        (doseq [[idx gpu] (map-indexed vector default-gpu-types)]
          (println (format "[%d] %s (%dGB)"
                           (inc idx)
                           (:displayName gpu)
                           (:memoryInGb gpu))))

        ;; Prompt for GPU selection
        (let [gpu-selection (prompt-number "\nSelect GPU type:" 1)
              selected-gpu (when (<= 1 gpu-selection (count default-gpu-types))
                             (nth default-gpu-types (dec gpu-selection)))

              ;; Configure pod details
              pod-name (prompt-with-default "Pod Name:" (:pod-name @config))
              gpu-count (prompt-number "GPU Count:" (:gpu-count @config))
              container-disk (prompt-number "Container Disk Size (GB):" (:container-disk @config))
              volume-size (prompt-number "Volume Size (GB):" (:volume-size @config))
              image (prompt-with-default "Docker Image:" (:image @config))
              ssh-key (prompt-with-default "SSH Key Path:" (:ssh-key @config))
              idle-timeout (prompt-number "Idle timeout before stopping pod (seconds):" (:pod-idle-timeout @config))]

          ;; Update config atom with user values
          (swap! config assoc
                 :pod-name pod-name
                 :gpu-count gpu-count
                 :container-disk container-disk
                 :volume-size volume-size
                 :image image
                 :ssh-key ssh-key
                 :pod-idle-timeout idle-timeout)

          ;; Return pod configuration
          {:gpu-type (:id selected-gpu)
           :gpu-type-name (:displayName selected-gpu)
           :gpu-count gpu-count
           :pod-name pod-name
           :container-disk container-disk
           :volume-size volume-size
           :image image}))

      ;; Normal flow when GPU types are successfully fetched
      (do
      ;; Display available GPU types
        (println "\nAvailable GPU Types:")
        (doseq [[idx gpu] (map-indexed vector gpu-types)]
          (println (format "[%d] %s (%dGB)"
                           (inc idx)
                           (:displayName gpu)
                           (:memoryInGb gpu))))

        ;; Prompt for GPU selection
        (let [gpu-selection (prompt-number "\nSelect GPU type:" 1)
              selected-gpu (when (<= 1 gpu-selection (count gpu-types))
                             (nth gpu-types (dec gpu-selection)))

              ;; Configure pod details
              pod-name (prompt-with-default "Pod Name:" (:pod-name @config))
              gpu-count (prompt-number "GPU Count:" (:gpu-count @config))
              container-disk (prompt-number "Container Disk Size (GB):" (:container-disk @config))
              volume-size (prompt-number "Volume Size (GB):" (:volume-size @config))
              image (prompt-with-default "Docker Image:" (:image @config))
              ssh-key (prompt-with-default "SSH Key Path:" (:ssh-key @config))
              idle-timeout (prompt-number "Idle timeout before stopping pod (seconds):" (:pod-idle-timeout @config))]

          ;; Update config atom with user values
          (swap! config assoc
                 :pod-name pod-name
                 :gpu-count gpu-count
                 :container-disk container-disk
                 :volume-size volume-size
                 :image image
                 :ssh-key ssh-key
                 :pod-idle-timeout idle-timeout)

          ;; Return pod configuration
          {:gpu-type (:id selected-gpu)
           :gpu-type-name (:displayName selected-gpu)
           :gpu-count gpu-count
           :pod-name pod-name
           :container-disk container-disk
           :volume-size volume-size
           :image image})))))

(defn wait-for-idle-and-stop
  "Waits for the specified idle timeout, then stops the pod.

   Args:
   api-key - RunPod API key
   pod-id - ID of the pod to stop
   idle-timeout - Time to wait in seconds before stopping

   Returns:
   true if pod was stopped, false otherwise"
  [api-key pod-id idle-timeout]
  (println "\n=== Waiting for idle timeout ===")
  (println (format "Pod will stop after %d seconds idle time..." idle-timeout))

  (let [countdown-fn (fn []
                       (dotimes [i idle-timeout]
                         (print (format "\rTime remaining: %d seconds  " (- idle-timeout i)))
                         (flush)
                         (Thread/sleep 1000))
                       (println "\nIdle timeout reached, stopping pod...")
                       true)]

    (let [_ (display-spinner "Waiting for idle timeout" countdown-fn)]
      (stop-pod api-key pod-id))))

(defn prepare-pod
  "Prepares a pod for execution, starting it if needed.

   Args:
   api-key - RunPod API key
   pod - Pod data from API

   Returns:
   Map with pod info and SSH config, or nil on failure"
  [api-key pod]
  (let [pod-id (:id pod)]
    (println "\n=== Preparing Pod for Execution ===")
    (println "Pod:" (:name pod))
    (println "Status:" (:desiredStatus pod))

    ;; Start pod if it's not running
    (when (not= "RUNNING" (:desiredStatus pod))
      (println "Pod is not running, starting it now...")
      (start-pod api-key pod-id))

    ;; Wait for pod to be running and SSH available
    (when-let [running-pod (wait-for-pod-running api-key pod-id)]
      (when-let [ready-pod (wait-for-ssh-port api-key pod-id)]
        (when-let [ssh-config (get-pod-ssh-details ready-pod)]
          {:pod ready-pod
           :ssh-config ssh-config})))))

(defn execute-workflow
  "Executes the full pod preparation, execution, and cleanup workflow.

   Args:
   api-key - RunPod API key
   md-file - Markdown file to process
   create-new-pod - Whether to create a new pod or use existing (boolean)

   Returns:
   Map with workflow execution details"
  [api-key md-file create-new-pod]
  (clear-screen)
  (println "=== Executing Workflow ===\n")
  (println "File:" md-file)

  ;; Prompt for output file
  (let [default-output (default-output-filename md-file)
        output-file (prompt-with-default "Output file:" default-output)]

    ;; Step 1: Get or create a pod
    (if create-new-pod
      ;; Create a new pod
      (let [pod-config (configure-pod api-key)]
        (println "\nPod Configuration:")
        (println (str "  GPU Type: " (:gpu-type-name pod-config)))
        (println (str "  GPU Count: " (:gpu-count pod-config)))
        (println (str "  Name: " (:pod-name pod-config)))
        (println "\nOutput Configuration:")
        (println (str "  Input file: " md-file))
        (println (str "  Output file: " output-file))

        (when (prompt-yn "\nProceed with pod creation?" true)
          ;; Step 2: Create pod
          (if-let [pod (create-pod api-key pod-config)]
            (let [pod-id (:id pod)]
              ;; Step 3: Wait for pod to start running
              (if-let [pod-info (prepare-pod api-key pod)]
                (let [ready-pod (:pod pod-info)
                      ssh-config (:ssh-config pod-info)]
                  ;; Step 4: Execute code blocks
                  (let [result (execute-site-gen md-file output-file ssh-config)]
                    ;; Step 5: Wait for idle timeout and stop pod
                    (wait-for-idle-and-stop api-key pod-id (:pod-idle-timeout @config))

                    ;; Return workflow result
                    {:success true
                     :pod-id pod-id
                     :execution-result result}))

;; Failed to prepare pod
                (do
                  (println "Failed to prepare pod. Cleaning up...")
                  (stop-pod api-key pod-id)
                  {:success false
                   :error "Failed to prepare pod"
                   :pod-id pod-id})))

            ;; Failed to create pod
            {:success false
             :error "Failed to create pod"}))

        ;; User aborted
        {:success false
         :cancelled true})

      ;; Use existing pod
      (let [pods (get-user-pods api-key)]
        (if (empty? pods)
          (do
            (println "No existing pods found. You need to create a new pod.")
            (execute-workflow api-key md-file true))

          (if-let [selected-pod (select-pod api-key pods)]
            (if (= selected-pod :create-new)
              ;; User wants to create a new pod after all
              (execute-workflow api-key md-file true)

              ;; Use the selected pod
              (let [pod-id (:id selected-pod)]
                (println "\nUsing existing pod:" (:name selected-pod))
                (println "Output Configuration:")
                (println (str "  Input file: " md-file))
                (println (str "  Output file: " output-file))

                (if (prompt-yn "\nProceed with execution?" true)
                  ;; Prepare pod (start if needed)
                  (if-let [pod-info (prepare-pod api-key selected-pod)]
                    (let [ready-pod (:pod pod-info)
                          ssh-config (:ssh-config pod-info)]
                      ;; Execute code blocks
                      (let [result (execute-site-gen md-file output-file ssh-config)]
                        ;; Ask if user wants to stop the pod
                        (when (prompt-yn "\nStop pod after execution?" false)
                          (stop-pod api-key pod-id))

                        ;; Return workflow result
                        {:success true
                         :pod-id pod-id
                         :execution-result result}))

                    ;; Failed to prepare pod
                    {:success false
                     :error "Failed to prepare pod"
                     :pod-id pod-id})

                  ;; User aborted
                  {:success false
                   :cancelled true})))

            ;; No pod selected
            {:success false
             :cancelled true}))))))

;;; ===================================================================
;;; === Main Entry Point ===
;;; ===================================================================

(defn -main
  "Main entry point for the RunPod Markdown Executor.

   Args:
   api-key - RunPod API key (required)
   [args] - Optional arguments:
     --create-pod/-c  - Force creation of a new pod
     --file/-f <file> - Markdown file to process

   Returns:
   nil"
  [api-key & args]
  (println "\n=== RunPod Markdown Executor ===")
  (println "Initializing...")

  (let [parsed-args (loop [remaining args
                           result {:create-pod false
                                   :file nil}]
                      (if (empty? remaining)
                        result
                        (let [arg (first remaining)]
                          (cond
                            (contains? #{"--create-pod" "-c"} arg)
                            (recur (rest remaining) (assoc result :create-pod true))

                            (contains? #{"--file" "-f"} arg)
                            (if (< (count remaining) 2)
                              (do
                                (println "Error: Missing filename after" arg)
                                result)
                              (recur (drop 2 remaining)
                                     (assoc result :file (second remaining))))

                            :else
                            (do
                              (println "Warning: Unknown argument" arg)
                              (recur (rest remaining) result))))))
        create-pod (:create-pod parsed-args)
        file (:file parsed-args)]

    (if file
      ;; If file was provided as argument, execute it directly
      (execute-workflow api-key file create-pod)
      ;; Otherwise, show the TUI menu
      (loop []
        (when-let [selected-file (select-markdown-file)]
          (execute-workflow api-key selected-file create-pod)
          (prompt "\nPress Enter to continue...")
          (recur))))))

;;; ===================================================================
;;; === Script Execution ===
;;; ===================================================================

(when (= *file* (System/getProperty "babashka.file"))
  (let [api-key (first *command-line-args*)]
    (if api-key
      (apply -main api-key (rest *command-line-args*))
      (println "Please provide your RunPod API key as an argument."))))

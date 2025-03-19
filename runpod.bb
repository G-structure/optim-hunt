#!/usr/bin/env bb

;;; === RunPod Manager ===
;;; A simplified command-line interface for managing RunPod resources
;;; Author: Assistant
;;; Date: 2024-03-19

(ns runpod.manager
  "Command-line tool for RunPod resource management.
   Provides a menu-driven interface to view and manage
   pods, check GPU pricing, and create new instances."
  (:require [babashka.http-client :as http]
            [cheshire.core :as json]
            [clojure.string :as str]))

;;; === Configuration ===

(def ^:private api-endpoint
  "RunPod's GraphQL API endpoint URL"
  "https://api.runpod.io/graphql")

;;; === API Queries ===

(def ^:private running-pods-query
  "GraphQL query to fetch all pods"
  "query Pods {
     myself {
       pods {
         id
         name
         podType
         desiredStatus
         costPerHr
         gpuCount
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
           gpus {
             id
             gpuUtilPercent
             memoryUtilPercent
           }
           container {
             cpuPercent
             memoryPercent
           }
         }
       }
     }
   }")

(def ^:private gpu-types-query
  "GraphQL query to fetch all GPU types and pricing information"
  "query GpuTypes {
     gpuTypes {
       id
       displayName
       memoryInGb
       secureCloud
       communityCloud
       securePrice
       communityPrice
       communitySpotPrice
       secureSpotPrice
       maxGpuCount
       maxGpuCountSecureCloud
       maxGpuCountCommunityCloud
     }
   }")

(def ^:private create-pod-mutation
  "GraphQL mutation to create a new pod"
  "mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
     podFindAndDeployOnDemand(input: $input) {
       id
       name
       imageName
       desiredStatus
       machineId
     }
   }")

(def ^:private create-spot-pod-mutation
  "GraphQL mutation to create a spot/interruptible pod"
  "mutation CreateSpotPod($input: PodRentInterruptableInput!) {
     podRentInterruptable(input: $input) {
       id
       name
       imageName
       desiredStatus
       machineId
     }
   }")

(def ^:private stop-pod-mutation
  "GraphQL mutation to stop a pod"
  "mutation StopPod($input: PodStopInput!) {
     podStop(input: $input) {
       id
       desiredStatus
     }
   }")

;;; === API Interaction Functions ===

(defn make-graphql-request
  "Makes an authenticated GraphQL request to RunPod API.

   Args:
     query - GraphQL query string
     api-key - RunPod API key for authentication
     variables - Optional variables map for GraphQL query

   Returns:
     Parsed JSON response on success, nil on failure"
  [query api-key & [variables]]
  (try
    (let [response (http/post api-endpoint
                              {:headers {"Content-Type" "application/json"
                                         "Authorization" (str "Bearer " api-key)}
                               :body (json/generate-string
                                      {:query query
                                       :variables (or variables {})})
                               :throw false})]
      (if (= 200 (:status response))
        (json/parse-string (:body response) true)
        (do
          (println "API Error:" (:status response))
          (when-let [errors (get-in (json/parse-string (:body response) true) [:errors])]
            (println "Error details:" (get-in errors [0 :message] "Unknown error")))
          nil)))
    (catch Exception e
      (println "Request failed:" (ex-message e))
      nil)))

(defn fetch-pods
  "Retrieves all pods for the authenticated user.

   Args:
     api-key - RunPod API key

   Returns:
     Vector of pod data maps"
  [api-key]
  (println "Fetching pods...")
  (when-let [response (make-graphql-request running-pods-query api-key)]
    (let [pods (get-in response [:data :myself :pods])]
      (println "Loaded" (count pods) "pods")
      pods)))

(defn fetch-gpu-types
  "Retrieves all GPU types and pricing information.

   Args:
     api-key - RunPod API key

   Returns:
     Vector of GPU type data maps"
  [api-key]
  (println "Fetching GPU types...")
  (when-let [response (make-graphql-request gpu-types-query api-key)]
    (let [gpu-types (get-in response [:data :gpuTypes])]
      (println "Loaded" (count gpu-types) "GPU types")
      gpu-types)))

(defn create-pod
  "Creates a new pod with specified parameters.

   Args:
     api-key - RunPod API key
     form - Map containing pod configuration parameters

   Returns:
     Created pod data on success, nil on failure"
  [api-key form]
  (let [{:keys [gpu-type gpu-count name image container-disk volume-size bid-mode bid-price]} form
        mutation (if bid-mode create-spot-pod-mutation create-pod-mutation)
        input-key :input
        variables {input-key (cond-> {:gpuTypeId gpu-type
                                      :gpuCount gpu-count
                                      :name name
                                      :imageName image
                                      :containerDiskInGb container-disk
                                      :volumeInGb volume-size
                                      :cloudType "ALL"
                                      :volumeMountPath "/workspace"
                                      :ports "8888/http,22/tcp"}
                               bid-mode (assoc :bidPerGpu bid-price))}]

    (println "Creating pod...")
    (when-let [response (make-graphql-request mutation api-key variables)]
      (if-let [errors (:errors response)]
        (println "Failed to create pod:" (get-in errors [0 :message] "Unknown error"))
        (let [result-key (if bid-mode :podRentInterruptable :podFindAndDeployOnDemand)
              pod (get-in response [:data result-key])]
          (println "Pod created successfully")
          pod)))))

(defn stop-pod
  "Stops a running pod.

   Args:
     api-key - RunPod API key
     pod-id - ID of the pod to stop

   Returns:
     true on success, false on failure"
  [api-key pod-id]
  (println "Stopping pod" pod-id "...")
  (when-let [response (make-graphql-request
                       stop-pod-mutation
                       api-key
                       {:input {:podId pod-id}})]
    (if-let [errors (:errors response)]
      (do
        (println "Failed to stop pod:" (get-in errors [0 :message] "Unknown error"))
        false)
      (do
        (println "Pod stopped successfully")
        true))))

;;; === Formatting and Display Functions ===

(defn clear-screen
  "Clears the terminal screen"
  []
  (print (str (char 27) "[2J" (char 27) "[H"))
  (flush))

(defn format-uptime
  "Formats uptime seconds into a human-readable string"
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

(defn format-pod-status
  "Returns a colored string for pod status"
  [status]
  (case status
    "RUNNING" "\033[32mRUNNING\033[0m"
    "EXITED" "\033[90mEXITED\033[0m"
    "TERMINATED" "\033[31mTERMINATED\033[0m"
    (str status)))

(defn format-pod-type
  "Returns a descriptive string for pod type"
  [pod-type]
  (case pod-type
    "RESERVED" "On-Demand"
    "INTERRUPTABLE" "Spot"
    (str pod-type)))

(defn format-cost
  "Formats cost value to dollars with 2 decimal places"
  [cost]
  (if cost
    (format "$%.2f" (double cost))
    "N/A"))

(defn display-pods-table
  "Displays pods in a tabular format with index numbers"
  [pods]
  (if (empty? pods)
    (println "No pods found.")
    (do
      (println (format "%-4s %-12s %-25s %-12s %-12s %-10s %-6s %-15s"
                       "NUM" "ID" "NAME" "STATUS" "TYPE" "COST/HR" "GPUs" "UPTIME"))
      (println (apply str (repeat 96 "-")))
      (doseq [[idx pod] (map-indexed vector pods)]
        (println (format "%-4d %-12s %-25s %-12s %-12s %-10s %-6d %-15s"
                         idx
                         (subs (:id pod) 0 (min 10 (count (:id pod))))
                         (subs (:name pod) 0 (min 25 (count (:name pod))))
                         (format-pod-status (:desiredStatus pod))
                         (format-pod-type (:podType pod))
                         (format-cost (:costPerHr pod))
                         (:gpuCount pod)
                         (format-uptime (get-in pod [:runtime :uptimeInSeconds]))))))))

(defn display-pod-details
  "Displays detailed information about a pod"
  [pod]
  (println "=== Pod Details ===")
  (println)
  (println (str "Name: " (:name pod)))
  (println (str "ID: " (:id pod)))
  (println (str "Status: " (format-pod-status (:desiredStatus pod))))
  (println (str "Type: " (format-pod-type (:podType pod))))
  (println (str "Cost/hr: " (format-cost (:costPerHr pod))))
  (println (str "GPU Count: " (:gpuCount pod)))
  (println (str "Uptime: " (format-uptime (get-in pod [:runtime :uptimeInSeconds]))))
  (println (str "Machine ID: " (:machineId pod)))

  (println)
  (println "GPU Usage:")
  (if-let [gpus (get-in pod [:runtime :gpus])]
    (doseq [gpu gpus]
      (println (str "  " (:id gpu) ": " (:gpuUtilPercent gpu) "% util, "
                    (:memoryUtilPercent gpu) "% mem")))
    (println "  No GPU data available"))

  (println)
  (println "Ports:")
  (if-let [ports (get-in pod [:runtime :ports])]
    (doseq [port ports]
      (println (str "  " (:type port) ": " (:publicPort port))))
    (println "  No port data available")))

(defn display-gpu-types-table
  "Displays GPU types in a tabular format"
  [gpu-types]
  (if (empty? gpu-types)
    (println "No GPU types found.")
    (do
      (println (format "%-4s %-25s %-7s %-12s %-12s %-10s"
                       "IDX" "GPU" "MEMORY" "ON-DEMAND" "SPOT" "MAX COUNT"))
      (println (apply str (repeat 80 "-")))
      (doseq [[idx gpu] (map-indexed vector gpu-types)]
        (let [secure-price (or (:securePrice gpu) (:communityPrice gpu))
              spot-price (or (:secureSpotPrice gpu) (:communitySpotPrice gpu))
              max-count (or (:maxGpuCountSecureCloud gpu)
                            (:maxGpuCountCommunityCloud gpu)
                            (:maxGpuCount gpu))]
          (println (format "%-4d %-25s %-7d %-12s %-12s %-10d"
                           idx
                           (:displayName gpu)
                           (:memoryInGb gpu)
                           (format-cost secure-price)
                           (format-cost spot-price)
                           max-count)))))))

;;; === User Interaction Functions ===

(defn prompt
  "Displays a prompt and reads a line of input"
  [message]
  (print message)
  (flush)
  (read-line))

(defn prompt-with-default
  "Displays a prompt with default value and reads input"
  [message default]
  (let [input (prompt (format "%s [%s]: " message default))]
    (if (str/blank? input) default input)))

(defn prompt-number
  "Prompts for a number with default value"
  [message default]
  (let [input (prompt (format "%s [%s]: " message default))]
    (if (str/blank? input)
      default
      (try
        (Integer/parseInt input)
        (catch Exception _
          (println "Invalid number, using default.")
          default)))))

(defn prompt-float
  "Prompts for a floating point number with default value"
  [message default]
  (let [input (prompt (format "%s [%s]: " message default))]
    (if (str/blank? input)
      default
      (try
        (Double/parseDouble input)
        (catch Exception _
          (println "Invalid number, using default.")
          default)))))

(defn prompt-yn
  "Prompts for a yes/no response"
  [message default]
  (let [default-str (if default "Y/n" "y/N")
        input (prompt (format "%s [%s]: " message default-str))]
    (if (str/blank? input)
      default
      (contains? #{"y" "Y" "yes" "Yes" "YES"} input))))

(defn select-from-list
  "Prompts user to select from a numbered list"
  [items formatter]
  (let [input (prompt "\nEnter selection number: ")]
    (try
      (let [idx (Integer/parseInt input)]
        (if (and (>= idx 0) (< idx (count items)))
          (nth items idx)
          (do
            (println "Invalid selection.")
            nil)))
      (catch Exception _
        (println "Please enter a valid number.")
        nil))))

(defn create-pod-form
  "Interactive form for creating a new pod"
  [api-key]
  (let [gpu-types (fetch-gpu-types api-key)]
    (if (empty? gpu-types)
      (println "No GPU types available. Cannot create pod.")
      (do
        (clear-screen)
        (println "=== Create New Pod ===\n")

        (println "Available GPU Types:")
        (display-gpu-types-table gpu-types)

        (when-let [selected-gpu (select-from-list gpu-types #(:displayName %))]
          (let [gpu-type (:id selected-gpu)
                gpu-name (:displayName selected-gpu)
                pod-name (prompt-with-default "Pod Name" "RunPod Instance")
                gpu-count (prompt-number "GPU Count" 1)
                container-disk (prompt-number "Container Disk Size (GB)" 40)
                volume-size (prompt-number "Volume Size (GB)" 100)
                image (prompt-with-default "Docker Image" "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
                bid-mode (prompt-yn "Use Spot/Interruptible pricing?" false)
                bid-price (when bid-mode
                            (prompt-float "Bid Price per GPU"
                                          (or (:secureSpotPrice selected-gpu)
                                              (:communitySpotPrice selected-gpu)
                                              0.5)))]

            (println "\nPod Configuration Summary:")
            (println (str "GPU Type: " gpu-name))
            (println (str "GPU Count: " gpu-count))
            (println (str "Pod Name: " pod-name))
            (println (str "Container Disk: " container-disk "GB"))
            (println (str "Volume Size: " volume-size "GB"))
            (println (str "Docker Image: " image))
            (println (str "Pricing: " (if bid-mode (str "Spot (Bid: " (format-cost bid-price) " per GPU)") "On-Demand")))

            (when (prompt-yn "\nConfirm creation?" true)
              (create-pod api-key {:gpu-type gpu-type
                                   :gpu-count gpu-count
                                   :name pod-name
                                   :image image
                                   :container-disk container-disk
                                   :volume-size volume-size
                                   :bid-mode bid-mode
                                   :bid-price bid-price}))))))))

(defn manage-pods
  "Menu for viewing and managing pods"
  [api-key]
  (loop []
    (clear-screen)
    (println "=== Manage Pods ===\n")
    (let [pods (fetch-pods api-key)]
      (display-pods-table pods)

      (println "\nOptions:")
      (println "1. Refresh pods list")
      (println "2. View pod details")
      (println "3. Stop a pod")
      (println "4. Return to main menu")

      (let [choice (prompt "\nEnter option: ")]
        (case choice
          "1" (do
                (println "Refreshing...")
                (recur))

          "2" (do
                (if (empty? pods)
                  (println "No pods available to view.")
                  (let [pod-num (prompt-number "Enter pod number to view details" 0)
                        pod (when (< pod-num (count pods)) (nth pods pod-num))]
                    (if pod
                      (do
                        (clear-screen)
                        (display-pod-details pod)
                        (prompt "\nPress Enter to continue..."))
                      (do
                        (println "Invalid pod number.")
                        (prompt "\nPress Enter to continue...")))))
                (recur))

          "3" (do
                (if (empty? pods)
                  (println "No pods available to stop.")
                  (let [pod-num (prompt-number "Enter pod number to stop" 0)
                        pod (when (< pod-num (count pods)) (nth pods pod-num))]
                    (if pod
                      (do
                        (println "Stopping pod:" (:name pod))
                        (when (prompt-yn "Are you sure you want to stop this pod?" false)
                          (stop-pod api-key (:id pod)))
                        (prompt "\nPress Enter to continue..."))
                      (do
                        (println "Invalid pod number.")
                        (prompt "\nPress Enter to continue...")))))
                (recur))

          "4" nil

          (do
            (println "Invalid option.")
            (prompt "\nPress Enter to continue...")
            (recur)))))))

(defn view-gpu-pricing
  "Display GPU pricing information"
  [api-key]
  (clear-screen)
  (println "=== GPU Pricing ===\n")
  (let [gpu-types (fetch-gpu-types api-key)]
    (display-gpu-types-table gpu-types)
    (prompt "\nPress Enter to return to main menu...")))

;;; === Main Menu ===

(defn main-menu
  "Main application menu"
  [api-key]
  (loop []
    (clear-screen)
    (println "=== RunPod Manager ===\n")
    (println "1. View and Manage Pods")
    (println "2. View GPU Pricing")
    (println "3. Create New Pod")
    (println "4. Exit")

    (let [choice (prompt "\nEnter option: ")]
      (case choice
        "1" (do (manage-pods api-key) (recur))
        "2" (do (view-gpu-pricing api-key) (recur))
        "3" (do (create-pod-form api-key) (recur))
        "4" (println "Goodbye!")
        (do
          (println "Invalid option.")
          (prompt "\nPress Enter to continue...")
          (recur))))))

;;; === Main Function ===

(defn -main
  "Main entry point for the application.

   Args:
     api-key - RunPod API key

   Side effects:
     Runs the interactive command-line interface"
  [api-key]
  (println "Initializing RunPod Manager...")
  (main-menu api-key))

;;; === Script Execution ===

(when (= *file* (System/getProperty "babashka.file"))
  (let [api-key (first *command-line-args*)]
    (if api-key
      (-main api-key)
      (println "Please provide your RunPod API key as an argument."))))

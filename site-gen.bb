#!/usr/bin/env bb

;;; ===================================================================
;;; === Site Generator Documentation ===
;;; ===================================================================
;;; This script generates HTML content from markdown files with special features:
;;; - Executes Python code blocks locally or remotely via SSH
;;; - Converts markdown to HTML using pandoc
;;; - Handles frontmatter metadata
;;; - Processes sidenotes
;;; - Caches code execution results
;;;
;;; The remote execution system:
;;; 1. Transfers the local source code to the remote machine
;;; 2. Syncs Python dependencies using uv
;;; 3. Executes code blocks with proper Python environment setup
;;; 4. Returns results for integration into the generated site

(require '[clojure.string :as str]
         '[clojure.java.shell :as shell]
         '[clojure.java.io :as io]
         '[cheshire.core :as json])

;;; ===================================================================
;;; === Configuration ===
;;; ===================================================================
(def template-file "template.html")
(def output-file "index.html")
(def cache-file ".code-outputs.json")
(def cache-dir (or (System/getenv "SITE_GEN_CACHE_DIR") ".cache"))  ; Configurable cache directory
(def temp-dir "/tmp/site-gen-remote")  ; Directory for temporary remote execution files
(def remote-src-dir (str temp-dir "/src"))  ; Directory for source code on remote server
(def remote-project-file (str temp-dir "/pyproject.toml"))  ; Path to remote pyproject.toml
(def default-remote-timeout 3600)  ; Default timeout for remote operations (1 hour)

;;; ===================================================================
;;; === Utility Functions ===
;;; ===================================================================

;; Sets the directory used for caching code block outputs
(defn set-cache-directory!
  "Sets the directory to use for caching code block outputs.
   Args:
     dir - Path to directory to use for cache files
   Returns:
     The new cache directory path
   Side effects:
     Updates global cache-dir variable"
  [dir]
  (alter-var-root #'cache-dir (constantly dir))
  cache-dir)

;; Sanitizes input strings to prevent shell injection
(defn sanitize-shell-input
  "Sanitizes input strings to prevent shell injection.
   Args:
     input - String to be sanitized for shell command use
   Returns:
     Safely escaped string for shell command use"
  [input]
  (when input
    (-> input
        (str/replace #"[\"']" "\\\\$0")      ; Escape quotes
        (str/replace #"\$" "\\\\$0")         ; Escape dollar signs
        (str/replace #"\`" "\\\\$0")         ; Escape backticks
        (str/replace #"\\" "\\\\$0")         ; Escape backslashes
        (str/replace #"[;&|<>]" "\\\\$0"))))  ; Escape shell special chars

;; Executes a function with a visual spinner to indicate progress
(defn show-spinner
  "Displays a spinning progress indicator in the console.
   Args:
     message - Text to display beside spinner
     task-fn - Function to execute while showing spinner
   Returns:
     Result of task-fn
   Side effects:
     Displays spinning animation in console during execution"
  [message task-fn]
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
      (let [result (task-fn)]
        (reset! running false)
        (deref spinner-thread 100 nil)
        (println "\r" message " Done!      ")
        result)
      (catch Exception e
        (reset! running false)
        (deref spinner-thread 100 nil)
        (println "\r" message " Failed!    ")
        (throw e)))))

;; Executes a function with an exclusive lock on a file
(defn with-file-lock
  "Executes a function with an exclusive lock on a file.
   Args:
     lock-file - Path to file to use for locking
     f - Function to execute with lock held
   Returns:
     Result of function f
   Side effects:
     Creates and deletes lock file"
  [lock-file f]
  (let [lock (io/file (str lock-file ".lock"))]
    (try
      ;; Try to create lock file
      (while (not (.createNewFile lock))
        (Thread/sleep 100))  ; Wait before retrying

      ;; Execute function with lock held
      (f)

      (finally
        ;; Always release the lock
        (when (.exists lock)
          (.delete lock))))))

;;; ===================================================================
;;; === SSH Connection Management ===
;;; ===================================================================

;; Stores SSH connection settings to avoid passing them through all functions
(def ssh-config (atom nil))

;; Tests the SSH connection to ensure it's properly configured
(defn test-ssh-connection
  "Tests SSH connection using provided configuration.
   Args:
     config - Map containing SSH connection details:
             :host - Remote server hostname/IP
             :port - SSH port number
             :user - SSH username
             :key  - Path to SSH private key file
   Returns:
     true if connection succeeds, false otherwise
   Side effects:
     Prints status messages about connection attempt"
  [config]
  (println "\n=== Testing SSH Connection ===")
  (println "Connecting to" (:host config) "on port" (:port config))
  (try
    (let [host (sanitize-shell-input (:host config))
          user (sanitize-shell-input (:user config))
          port (sanitize-shell-input (:port config))
          key (sanitize-shell-input (:key config))
          cmd (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "  ; SSH command with security options
                   (when key
                     (str "-i " key " "))                     ; Add identity file if specified
                   "-p " port " "                             ; Specify port number
                   user "@" host " "                          ; User and host address
                   "echo 'SSH connection successful'")        ; Simple test command
          result (shell/sh "bash" "-c" cmd)]                  ; Execute SSH command
      (if (zero? (:exit result))                              ; Check exit code
        (do
          (println "SSH connection established successfully")
          true)                                               ; Return success
        (do
          (println "SSH connection failed:" (:err result))
          false)))                                            ; Return failure
    (catch Exception e                                        ; Handle any connection errors
      (println "SSH connection error:" (str e))
      false)))                                                ; Return failure on exception

;; Sets up the remote environment with source code and dependencies
(defn setup-remote-environment
  "Sets up the remote execution environment on the SSH server.
   This comprehensive setup includes:
   1. Installing uv if not already present
   2. Creating necessary directories
   3. Transferring the local src/ directory
   4. Transferring pyproject.toml
   5. Transferring README.md
   6. Installing dependencies with uv sync

   Args:
     config - Map containing SSH connection details

   Returns:
     true if setup succeeds, false otherwise

   Side effects:
     Creates remote directories
     Transfers source code files
     Installs Python dependencies
     Prints status messages about environment setup"
  [config]
  (println "\n=== Setting Up Remote Environment ===")
  (try
    ;; Step 0: Install uv if not present
    (println "Ensuring uv is installed on remote server...")
    (let [host (sanitize-shell-input (:host config))
          user (sanitize-shell-input (:user config))
          port (sanitize-shell-input (:port config))
          key (sanitize-shell-input (:key config))
          uv-install-cmd (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
                              (when key
                                (str "-i " key " "))
                              "-p " port " "
                              user "@" host " "
                              "\"if command -v uv >/dev/null 2>&1 || [ -x /root/.local/bin/uv ]; then "
                              "echo 'uv already installed'; "
                              "else "
                              "curl -LsSf https://astral.sh/uv/install.sh | sh && echo 'uv installed successfully'; "
                              "fi\"")
          uv-install-result (shell/sh "bash" "-c" uv-install-cmd)]

      (when-not (zero? (:exit uv-install-result))
        (println "Failed to install uv on remote server:" (:err uv-install-result))
        (throw (Exception. "Failed to install uv on remote server"))))

    ;; Step 1: Create directories on remote server
    (let [host (sanitize-shell-input (:host config))
          user (sanitize-shell-input (:user config))
          port (sanitize-shell-input (:port config))
          key (sanitize-shell-input (:key config))
          create-dir-cmd (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "  ; SSH with security options
                              (when key
                                (str "-i " key " "))                     ; Add identity file if specified
                              "-p " port " "                             ; Specify port number
                              user "@" host " "                          ; User and host address
                              "\"mkdir -p " temp-dir " && "              ; Create temp directory
                              "mkdir -p " remote-src-dir " && "          ; Create source directory
                              "echo 'Remote directories created'\"")     ; Confirmation message
          create-result (shell/sh "bash" "-c" create-dir-cmd)]           ; Execute directory creation

      (when-not (zero? (:exit create-result))                           ; Check for errors
        (println "Remote directory creation failed:" (:err create-result))
        (throw (Exception. "Failed to create remote directories"))))

    ;; Step 2: Transfer local src/ directory to remote server
    (println "Transferring source code to remote server...")
    (when (.exists (io/file "src"))                                     ; Check if src exists locally
      (let [host (sanitize-shell-input (:host config))
            user (sanitize-shell-input (:user config))
            port (sanitize-shell-input (:port config))
            key (sanitize-shell-input (:key config))
            scp-src-cmd (str "scp -o BatchMode=yes -o StrictHostKeyChecking=no "  ; SCP with security options
                             (when key
                               (str "-i " key " "))                     ; Add identity file if specified
                             "-P " port " -r "                          ; Port and recursive options
                             "src/* "                                   ; Source directory contents
                             user "@" host ":" remote-src-dir)          ; Destination path
            scp-src-result (shell/sh "bash" "-c" scp-src-cmd)]          ; Execute file transfer

        (when-not (zero? (:exit scp-src-result))                        ; Check for transfer errors
          (println "Source code transfer failed:" (:err scp-src-result))
          (throw (Exception. "Failed to transfer source code to remote server")))))

    ;; Step 3: Transfer pyproject.toml to remote server
    (println "Transferring pyproject.toml to remote server...")
    (when (.exists (io/file "pyproject.toml"))                          ; Check if pyproject.toml exists
      (let [host (sanitize-shell-input (:host config))
            user (sanitize-shell-input (:user config))
            port (sanitize-shell-input (:port config))
            key (sanitize-shell-input (:key config))
            scp-toml-cmd (str "scp -o BatchMode=yes -o StrictHostKeyChecking=no " ; SCP with security options
                              (when key
                                (str "-i " key " "))                    ; Add identity file if specified
                              "-P " port " "                            ; Port option
                              "pyproject.toml "                         ; Source file
                              user "@" host ":" remote-project-file)    ; Destination path
            scp-toml-result (shell/sh "bash" "-c" scp-toml-cmd)]        ; Execute file transfer

        (when-not (zero? (:exit scp-toml-result))                      ; Check for transfer errors
          (println "pyproject.toml transfer failed:" (:err scp-toml-result))
          (throw (Exception. "Failed to transfer pyproject.toml to remote server")))))

    ;; Step 4: Transfer README.md to remote server
    (println "Transferring README.md to remote server...")
    (when (.exists (io/file "README.md"))                               ; Check if README.md exists
      (let [host (sanitize-shell-input (:host config))
            user (sanitize-shell-input (:user config))
            port (sanitize-shell-input (:port config))
            key (sanitize-shell-input (:key config))
            remote-readme-file (str temp-dir "/README.md")              ; Path to remote README.md
            scp-readme-cmd (str "scp -o BatchMode=yes -o StrictHostKeyChecking=no " ; SCP with security options
                                (when key
                                  (str "-i " key " "))                   ; Add identity file if specified
                                "-P " port " "                           ; Port option
                                "README.md "                             ; Source file
                                user "@" host ":" remote-readme-file)    ; Destination path
            scp-readme-result (shell/sh "bash" "-c" scp-readme-cmd)]    ; Execute file transfer

        (when-not (zero? (:exit scp-readme-result))                     ; Check for transfer errors
          (println "README.md transfer failed:" (:err scp-readme-result))
          (throw (Exception. "Failed to transfer README.md to remote server")))))

    ;; Step 5: Install dependencies with uv sync
    (println "Installing Python dependencies on remote server...")
    (let [host (sanitize-shell-input (:host config))
          user (sanitize-shell-input (:user config))
          port (sanitize-shell-input (:port config))
          key (sanitize-shell-input (:key config))
          install-cmd (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "    ; SSH with security options
                           (when key
                             (str "-i " key " "))                       ; Add identity file if specified
                           "-p " port " "                               ; Port option
                           user "@" host " "                            ; User and host address
                           "\"cd " temp-dir " && "                      ; Change to temp directory
                           "export PATH=/root/.local/bin:$PATH && "     ; Ensure uv is in PATH
                           "uv sync && "                                ; Install dependencies with uv
                           "echo 'Python dependencies installed successfully'\"") ; Confirmation message
          install-result (shell/sh "bash" "-c" install-cmd)]            ; Execute dependency installation

      (if (zero? (:exit install-result))                                ; Check for installation success
        (do
          (println "Remote environment setup complete:")
          (println (:out install-result))
          true)                                                         ; Return success
        (do
          (println "Failed to install Python dependencies:" (:err install-result))
          false)))                                                      ; Return failure

    (catch Exception e                                                  ; Handle any setup errors
      (println "Remote environment setup error:" (str e))
      false)))                                                          ; Return failure on exception                                                       ; Return failure on exception

;; Cleans up the remote environment after execution
(defn cleanup-remote-environment
  "Cleans up temporary files and directories created on the remote server.
   Args:
     config - Map containing SSH connection details
   Returns:
     true if cleanup succeeds, false otherwise
   Side effects:
     Removes temporary directories on remote server"
  [config]
  (println "\n=== Cleaning Up Remote Environment ===")
  (try
    (let [host (sanitize-shell-input (:host config))
          user (sanitize-shell-input (:user config))
          port (sanitize-shell-input (:port config))
          key (sanitize-shell-input (:key config))
          cleanup-cmd (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
                           (when key
                             (str "-i " key " "))
                           "-p " port " "
                           user "@" host " "
                           "\"rm -rf " temp-dir " && echo 'Remote cleanup complete'\"")
          cleanup-result (shell/sh "bash" "-c" cleanup-cmd)]

      (if (zero? (:exit cleanup-result))
        (do
          (println "Remote environment cleanup successful")
          true)
        (do
          (println "Remote environment cleanup failed:" (:err cleanup-result))
          false)))
    (catch Exception e
      (println "Remote environment cleanup error:" (str e))
      false)))

;; Executes code on the remote server via SSH
(defn execute-remote-python
  "Executes Python code on a remote server through SSH.
   Uses uv run to execute code with the properly configured Python environment.
   Ensures PYTHONPATH includes the transferred source directory.

   Args:
     code   - String containing Python source code to execute remotely
     config - Map containing SSH connection details
     timeout - Optional timeout in seconds (default: 1 hour)

   Returns:
     Map containing execution results:
       :exit - Exit code (0 for success)
       :out  - Standard output from execution
       :err  - Standard error from execution

   Side effects:
     Transfers code to remote server
     Executes code remotely
     Prints status messages about remote execution"
  [code config & {:keys [timeout] :or {timeout default-remote-timeout}}]
  (show-spinner "Executing code remotely"
                (fn []
                  (let [;; Create unique filename for this execution using timestamp and hash
                        timestamp (System/currentTimeMillis)                                  ; Current time for uniqueness
                        code-hash (hash code)                                                 ; Hash of code for uniqueness
                        remote-file (str temp-dir "/code-" timestamp "-" code-hash ".py")     ; Remote file path

               ;; Write code to temporary local file
                        local-file (str "/tmp/code-" timestamp "-" code-hash ".py")]          ; Local temporary file path

                    (try
                      (spit local-file code)                                                  ; Write code to local file

             ;; Transfer file to remote server
                      (let [host (sanitize-shell-input (:host config))
                            user (sanitize-shell-input (:user config))
                            port (sanitize-shell-input (:port config))
                            key (sanitize-shell-input (:key config))
                            scp-cmd (str "scp -o BatchMode=yes -o StrictHostKeyChecking=no "  ; SCP with security options
                                         (when key
                                           (str "-i " key " "))                               ; Add identity file if specified
                                         "-P " port " "                                       ; Port option
                                         local-file " "                                       ; Source file
                                         user "@" host ":" remote-file)                       ; Destination path
                            _ (println "Transferring code to remote server")
                            scp-result (shell/sh "bash" "-c" scp-cmd)]                        ; Execute file transfer

               ;; If file transfer succeeded, execute code remotely
                        (when-not (zero? (:exit scp-result))                                  ; Check transfer status
                          (println "Failed to transfer code file:" (:err scp-result)))

                        (if (zero? (:exit scp-result))                                        ; Proceed only if transfer succeeded
                          (let [;; Execute Python on remote server with uv run and proper PYTHONPATH
                                exec-cmd (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "  ; SSH with security options
                                              (when key
                                                (str "-i " key " "))                          ; Add identity file if specified
                                              "-p " port " "                                  ; Port option
                                              user "@" host " "                               ; User and host address
                                              "\"cd " temp-dir " && "                         ; Change to temp directory
                                              "timeout " timeout "s "                         ; Add timeout to command
                                              "bash -c 'export PATH=/root/.local/bin:$PATH && "  ; Start bash with proper PATH
                                              "export PYTHONPATH=" remote-src-dir " && "      ; Set Python path as env variable
                                              "/root/.local/bin/uv run python3 " remote-file "'\"")  ; Use full path to uv
                                _ (println "Executing Python code remotely")
                                result (try
                                         (shell/sh "bash" "-c" exec-cmd)                      ; Execute Python code remotely
                                         (catch Exception e
                                           (println "Remote execution failed with exception:" (str e))
                                           {:exit 1
                                            :out ""
                                            :err (str "Remote execution error: " (str e))}))]

                   ;; Report error details if execution failed
                            (when-not (zero? (:exit result))
                              (println "Remote execution failed with exit code" (:exit result))
                              (println "Error message:" (:err result)))

                   ;; Clean up remote and local temporary files
                            (shell/sh "bash" "-c"                                             ; Execute cleanup command
                                      (str "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "  ; SSH with security options
                                           (when key
                                             (str "-i " key " "))                             ; Add identity file if specified
                                           "-p " port " "                                     ; Port option
                                           user "@" host " "                                  ; User and host address
                                           "\"rm -f " remote-file "\""))                      ; Remove remote file

                   ;; Return execution results
                            result)                                                           ; Return execution result map

                 ;; Return error if transfer failed
                          {:exit 1                                                            ; Error exit code
                           :out ""                                                            ; Empty output
                           :err (str "Failed to transfer code to remote server: " (:err scp-result))}))
                      (finally
                        (io/delete-file local-file true)))))))                        ; Always clean up local file

;;; ===================================================================
;;; === Cache Management Functions ===
;;; ===================================================================

;; Ensures the existence of a cache directory for storing block execution outputs
(defn ensure-cache-dir!
  "Manages the cache directory for storing code block execution results.
   Args:
     None
   Returns:
     true if directory exists and is accessible, false otherwise
   Side effects:
     Creates cache directory if it doesn't exist
     Prints status messages about directory creation"
  []
  (let [dir (io/file cache-dir)]               ; Create file object for cache directory
    (when-not (.exists dir)                    ; Check if directory needs to be created
      (println "Creating cache directory:" cache-dir)
      (.mkdir dir))                            ; Attempt directory creation if missing
    (.exists dir)))                            ; Return existence status

;; Constructs filesystem path for code block cache files based on input file and block ID
(defn cache-file-path
  "Builds standardized filesystem path for storing code block execution results.
   Args:
     input-file - Path to source markdown file containing the code block
     block-id   - Unique numeric identifier for the specific code block
   Returns:
     String containing full path to cache file in format:
     {cache-dir}/{input-filename}-block-{id}.edn"
  [input-file block-id]
  (let [base-name (-> input-file
                      io/file                            ; Convert to File object
                      .getName                           ; Extract filename
                      (str/replace #"\.[^.]+$" ""))]     ; Strip file extension
    (str cache-dir                                       ; Construct path:
         "/"                                             ; 1. Start with cache dir
         base-name                                       ; 2. Add base filename
         "-block-"                                       ; 3. Add block identifier
         block-id                                        ; 4. Add block ID
         ".edn")))                                       ; 5. Add EDN extension

;; Retrieves previously cached execution output for a specific code block
(defn load-block-cache
  "Loads and parses cached execution output for an individual code block.
   Args:
     input-file - Path to source markdown file containing the block
     block-id   - Unique numeric identifier for the code block
   Returns:
     String containing cached output if available and valid,
     nil if cache doesn't exist or can't be read.
   Side effects:
     Ensures cache directory exists
     Prints status messages about cache operations"
  [input-file block-id]
  (println "\n=== Loading Block Cache ===")
  (ensure-cache-dir!)                                     ; Make sure cache directory exists
  (let [cache-path (cache-file-path input-file block-id)] ; Get path for this block's cache
    (println "Checking cache file:" cache-path)
    (when (.exists (io/file cache-path))                  ; Only proceed if cache exists
      (try
        (let [cached-data (read-string (slurp cache-path))]  ; Parse EDN cache format
          (println "Found cached output for block:" block-id)
          (:output cached-data))                             ; Extract just the output portion
        (catch Exception e                                   ; Handle any parsing errors
          (println "Error reading cache for block" block-id ":" (str e))
          nil)))))                                           ; Return nil on error

;; Loads and assembles all cached code block outputs from individual EDN files
;; Files are found by matching pattern: {input-filename}-block-{id}.edn
(defn load-cache
  "Loads all cached code block execution outputs for a given markdown file.
   Args:
     input-file - Path to source markdown file containing code blocks
   Returns:
     Map where:
       - Keys are block ID strings from cache filenames
       - Values are maps containing full cache data (:code, :output, etc)
     Returns empty map if no valid cache files found"
  [input-file]
  (println "\n=== Loading Full Cache ===")
  (ensure-cache-dir!)
  (let [;; Build regex pattern to match cache files for this input
        cache-pattern (re-pattern (str (-> input-file
                                           io/file                            ; Convert to File object
                                           .getName                           ; Get filename
                                           (str/replace #"\.[^.]+$" ""))      ; Strip extension
                                       "-block-\\d+\\.edn"))                  ; Match block ID pattern
        ;; Find all matching cache files in cache directory
        cache-files (filter #(re-matches cache-pattern (.getName %))
                            (.listFiles (io/file cache-dir)))]
    (println "Found" (count cache-files) "cache files")
    (into {}
          (for [cache-file cache-files
                ;; Extract block ID from filename and load cache data
                :let [block-id (second (re-find #"-block-(\d+)\.edn$" (.getName cache-file)))
                      cached-data (try
                                    (read-string (slurp cache-file))          ; Parse EDN format
                                    (catch Exception e nil))]                 ; Return nil on error
                :when cached-data]                                            ; Skip invalid cache entries
            [block-id cached-data]))))                                        ; Map ID to full cache data

;; Loads targeted cache data for specific block or all blocks
(defn load-targeted-cache
  "Loads only the specified block's cache if computing a single block.
   Args:
     input-file - Path to source markdown file
     block-id - ID of block to load cache for, or nil for all blocks
   Returns:
     Map of cached block data, either for single block or all blocks"
  [input-file block-id]
  (if block-id
    ;; Load just the specified block cache
    (let [cache-data (load-block-cache input-file block-id)]
      (when cache-data
        {block-id {:output cache-data}}))
    ;; Load all cached blocks
    (load-cache input-file)))                                        ; Map ID to full cache data

;;; ===================================================================
;;; === Cache Writing Functions ===
;;; ===================================================================

;; Writes execution results and metadata for a single code block to cache file
(defn save-block-cache
  "Persists code block execution output and metadata to a cache file.
   Args:
     input-file - Path to source markdown file containing the code block
     block-id   - Unique numeric identifier for the specific code block
     code       - Source code string that was executed
     output     - Resulting output string from code execution
   Returns:
     nil, but writes cache file as side effect
   Side effects:
     Creates cache directory if needed
     Writes EDN format cache file with execution data"
  [input-file block-id code output]
  (println "\n=== Saving Block Cache ===")
  (ensure-cache-dir!)                                       ; Ensure cache directory exists
  (let [cache-path (cache-file-path input-file block-id)    ; Get path for this block's cache
        ;; Construct cache data structure with metadata
        cache-data {:block-id block-id                      ; Store block identifier
                    :code code                              ; Store original source code
                    :output output                          ; Store execution output
                    :timestamp (System/currentTimeMillis)}] ; Add timestamp for cache invalidation
    (println "Saving cache for block" block-id "to:" cache-path)
    (with-file-lock cache-path
      #(spit cache-path (pr-str cache-data)))))            ; Use file locking to prevent conflicts

;; Writes all computed code block outputs to their respective cache files
(defn save-cache
  "Persists all code block execution outputs to their individual cache files.
   Args:
     input-file - Path to source markdown file containing the blocks
     cache-map  - Map where keys are block IDs and values are execution outputs
     blocks     - Sequence of maps containing parsed code block definitions:
                 :id - Block identifier
                 :code - Source code for the block
   Returns:
     nil
   Side effects:
     Creates/updates cache files for each block that has output"
  [input-file cache-map blocks]
  (println "\n=== Saving Full Cache ===")
  (doseq [block blocks                                  ; Iterate through all code blocks
          :let [block-id (:id block)                    ; Extract block ID
                output (get cache-map block-id)]        ; Get cached output for this block
          :when output]                                 ; Only process blocks with output
    (save-block-cache input-file block-id (:code block) output)))  ; Write to individual cache file

;;; ===================================================================
;;; === Code Block Processing ===
;;; ===================================================================

;; Formats source code into syntax-highlighted HTML blocks with proper escaping
(defn wrap-code-block
  "Wraps source code in HTML structure for syntax highlighting display.
   Args:
     code - The source code string to be formatted
     lang - Programming language identifier for syntax highlighting
   Returns:
     HTML string containing wrapped and escaped code with proper highlighting classes"
  [code lang]
  (format "<div class=\"code-block\"><pre class=\"line-numbers\"><code class=\"language-%s\">%s</code></pre></div>"
          lang                                          ; Language class for syntax highlighter
          (-> code
              (str/replace "<" "&lt;")                  ; Escape < to prevent HTML injection
              (str/replace ">" "&gt;"))))               ; Escape > to prevent HTML injection

;; Handles the transformation of markdown code blocks into appropriate HTML or preserved format
(defn process-code-blocks
  "Processes markdown-style code blocks, either preserving executable blocks or converting to HTML.
   Args:
     content - String containing markdown with ```language code``` blocks
   Returns:
     String with code blocks either preserved (if executable) or converted to syntax-highlighted HTML"
  [content]
  (println "\n=== Processing Code Blocks ===")
  (let [;; Define regex pattern to match code blocks:
        ;; (?ms)     - Enable multiline and dotall mode
        ;; ```(\w+)  - Match opening fence and capture language identifier
        ;; \n(.*?)\n - Capture code content between newlines
        ;; ```       - Match closing fence
        code-block-pattern #"(?ms)```(\w+)\n(.*?)\n```"]

    ;; Process each code block match in the content
    (str/replace content code-block-pattern
                 (fn [[_ lang code]]                      ; Destructure match into language and code
                   (if (str/includes? content "<<execute")
                     (str "```" lang "\n" code "\n```")   ; Preserve executable blocks as-is
                     (wrap-code-block code lang))))))     ; Convert regular blocks to HTML

;;; ===================================================================
;;; === Code Execution ===
;;; ===================================================================

;; Validates whether a string contains HTML markup by checking for common HTML indicators
(defn is-html?
  "Determines if a string contains HTML content by checking for opening tags.
   Args:
     s - String to check for HTML content
   Returns:
     true if string appears to be HTML, false otherwise"
  [s]
  (or (str/starts-with? (str/trim s) "<")              ; Check for standard HTML tag
      (str/starts-with? (str/trim s) "<!DOCTYPE")))    ; Check for DOCTYPE declaration

;; Executes Python code locally or remotely and wraps its output in appropriate HTML formatting
(defn run-python
  "Executes Python code locally or remotely and formats the output as HTML.
   Args:
     code - String containing Python source code to execute
   Returns:
     HTML-formatted string containing either:
     - Execution output wrapped in <pre> tags
     - Raw HTML output if result is already HTML
     - Error message wrapped in error-styled <pre> tags
   Side effects:
     Executes code on local or remote Python interpreter
     Prints status messages about execution"
  [code]
  (println "\n=== Executing Python Code ===")
  (println "Code to execute:\n" code)
  (try
    (let [;; Determine if we should execute remotely based on ssh-config
          remote? (and @ssh-config
                       (test-ssh-connection @ssh-config))

          ;; Execute either locally or remotely based on configuration
          res (if remote?
                (do
                  (println "Executing via SSH on" (:host @ssh-config))
                  (execute-remote-python code @ssh-config))
                (do
                  (println "Executing locally")
                  (shell/sh "python3" "-c" code)))]

      ;; Process result regardless of execution method
      (if (zero? (:exit res))
        (let [output (:out res)]
          (println "Execution successful. Output length:" (count output))
          (if (is-html? output)
            output
            (str "<pre class=\"code-output\">" output "</pre>")))
        (do
          (println "Error running python code:" (:err res))
          (str "<pre class=\"code-error\">" (:err res) "</pre>"))))
    (catch Exception e
      (println "Exception running Python code:")
      (println "  - Error type:" (.getName (.getClass e)))
      (println "  - Message:" (.getMessage e))
      (println "  - Hint: Check that Python and required libraries are installed")
      (str "<pre class=\"code-error\">Error: " (str e)
           "\n\nHint: Ensure Python 3 is installed and available in your PATH</pre>"))))

;;; ===================================================================
;;; === Markdown Processing ===
;;; ===================================================================

;; Transforms markdown content to HTML using pandoc with support for math expressions
(defn markdown->html
  "Converts markdown text to HTML format using pandoc with LaTeX math support.
   Args:
     markdown - String containing markdown content to convert
   Returns:
     HTML string if conversion succeeds, original markdown string if conversion fails"
  [markdown]
  (println "\n=== Converting Markdown to HTML ===")
  (let [res (shell/sh "pandoc"                           ; Execute pandoc command
                      "--from=markdown+tex_math_dollars+tex_math_single_backslash"  ; Enable math syntax
                      "--to=html"                        ; Output HTML format
                      "--mathjax"                        ; Add MathJax support
                      "--wrap=none"                      ; Disable text wrapping
                      :in markdown)]                     ; Input markdown content
    (if (zero? (:exit res))                              ; Check if conversion succeeded
      (do
        (println "Conversion successful")
        (:out res))                                      ; Return converted HTML
      (do (println "Pandoc error:" (:err res))
          markdown))))                                   ; Return original on error

;; Transforms special markdown sidenote syntax into HTML div elements with proper formatting
(defn process-sidenotes
  "Processes markdown content containing ^^^...^^^ sidenote markers.
   Args:
     content - String containing markdown with sidenote markers
   Returns:
     String with sidenotes converted to HTML div elements with 'sidenote' class"
  [content]
  (println "\n=== Processing Sidenotes ===")
  (let [sidenote-pattern #"(?s)\^\^\^(.*?)\^\^\^"       ; Pattern to match ^^^content^^^ with any chars
        matches (re-seq sidenote-pattern content)]      ; Find all sidenote matches in content
    (println "Found" (count matches) "sidenotes")
    (let [;; Function to process single sidenote match
          process-single-sidenote (fn [[_ note-content]]
                                    (let [processed-content (markdown->html note-content)] ; Convert note content
                                      (format "<div class=\"sidenote\">%s</div>"           ; Wrap in div
                                              (str/trim processed-content))))]
      (str/replace content sidenote-pattern process-single-sidenote)))) ; Replace all matches

;;; ===================================================================
;;; === Block Parsing and Execution ===
;;; ===================================================================

;; Locates and parses executable code blocks from content, validating block IDs
(defn find-code-blocks
  "Extracts executable Python code blocks from content and validates their uniqueness.
   Args:
     content - String containing executable code block markers in the format:
              <<execute id=\"ID\" output=\"TYPE\">>```python\nCODE\n```<</execute>>
   Returns:
     Sequence of maps containing parsed block details:
       :id          - Unique block identifier
       :output-type - Desired output format
       :code        - Python source code
       :full-match  - Complete matched block text
   Throws:
     Exception if duplicate block IDs are found"
  [content]
  (println "\n=== Finding Code Blocks ===")
  (let [;; Pattern to match executable code blocks with metadata
        block-pattern #"(?ms)<<execute\s+id=\"(\d+)\"\s+output=\"([^\"]+)\">>\s*```python\n(.*?)\n```\s*<<\/execute>>"
        matches (re-seq block-pattern content)]         ; Find all code blocks in content
    (println "Found" (count matches) "code blocks")
    (let [blocks (map (fn [[full-match id output-type code]]  ; Transform matches into structured maps
                        {:id id
                         :output-type output-type
                         :code (str/trim code)          ; Clean up code whitespace
                         :full-match full-match})
                      matches)]
      (let [ids (map :id blocks)                        ; Extract all block IDs
            ;; Find any IDs that appear more than once
            duplicates (keep (fn [[id freq]] (when (> freq 1) id))
                             (frequencies ids))]
        (when (seq duplicates)                          ; Throw error if duplicates found
          (throw (Exception. (str "Duplicate block IDs found: " (str/join ", " duplicates)))))
        blocks))))                                      ; Return processed blocks

;;; ===================================================================
;;; === Output Processing ===
;;; ===================================================================

;; Processes code execution output by either returning it raw or converting through pandoc
(defn process-output
  "Processes code block execution output based on specified output type.
   Args:
     output      - String containing the raw execution output
     output-type - Format to process output into ('raw' or 'pandoc')
   Returns:
     String containing either raw output or pandoc-converted HTML.
   Throws:
     Exception if output-type is invalid"
  [output output-type]
  (println "\n=== Processing Output ===")
  (println "Processing output of type:" output-type)
  (case output-type
    "raw" (do
            (println "Using raw output")                ; Pass through unchanged
            output)
    "pandoc" (do
               (println "Converting output through pandoc")
               (let [res (shell/sh "pandoc"             ; Process through pandoc
                                   "--from=markdown"    ; Input markdown format
                                   "--to=html"          ; Output HTML format
                                   :in output)]
                 (if (zero? (:exit res))                ; Check if conversion succeeded
                   (:out res)                           ; Return converted HTML
                   (do
                     (println "Pandoc error:" (:err res))
                     output))))                         ; Return original on error
    (throw (Exception. (str "Invalid output type: " output-type)))))

;;; ===================================================================
;;; === Main Processing Functions ===
;;; ===================================================================

;; Checks if a block should be computed based on cache and configuration
(defn should-compute-block?
  "Determines if a code block should be computed based on cache and flags.
   Args:
     block      - Code block map containing :id and :code
     cache-data - Full cached data for the block (if any)
     compute-id - Optional ID of block to compute
     recompute? - Flag to force recomputation
   Returns:
     true if block should be computed, false if cache should be used"
  [block cache-data compute-id recompute?]
  (let [block-id (:id block)]
    (do
      (println "Checking block" block-id)
      (println "Cache data present:" (boolean cache-data))
      (println "Code matches:" (= (:code cache-data) (:code block)))
      (or recompute?                                    ; Recompute if forced
          (= block-id compute-id)                       ; Recompute if specifically requested
          (not cache-data)                              ; Compute if no cache exists
          (not= (:code cache-data) (:code block))))))   ; Compute if code changed

;; Core function that handles Python code block processing workflow:
;; 1. Loads/manages execution cache
;; 2. Executes or retrieves cached block outputs
;; 3. Integrates results into document content
(defn execute-blocks
  "Processes Python code blocks with caching and content integration.
   Args:
     input-file  - Path to markdown file containing code blocks
     content     - String containing markdown content with code blocks
     compute-id  - Optional ID of specific block to recompute
     recompute?  - Flag to force recomputation of all blocks
   Returns:
     String containing content with executed code blocks and outputs integrated"
  [input-file content compute-id recompute?]
  (println "\n=== Executing Blocks ===")
  (when compute-id
    (println "Computing only block with ID:" compute-id))
  (when recompute?
    (println "Forcing recomputation of all blocks"))

  (let [existing-cache (load-targeted-cache input-file compute-id) ; Load optimized cache data
        execution-cache (atom {})                                  ; Initialize cache for this run
        blocks (find-code-blocks content)]                         ; Extract code blocks from content

    ;;; --- Process Each Code Block ---
    (doseq [block blocks]
      (println "\nProcessing block" (:id block))
      (let [cached-data (get existing-cache (:id block))  ; Get any existing cached output
            needs-compute? (should-compute-block? block cached-data compute-id recompute?)]

        ;; Either compute new output or use cache
        (if needs-compute?
          (do
            (println "Computing block" (:id block))
            (let [output (run-python (:code block))       ; Execute Python code
                  processed (process-output output (:output-type block))]        ; Format output
              (save-block-cache input-file (:id block) (:code block) processed)  ; Cache result
              (swap! execution-cache assoc (:id block) processed))) ; Store in current cache
          (do
            (println "Using cached output for block" (:id block))
            (swap! execution-cache assoc (:id block) (:output cached-data))))))  ; Use cached output

    ;;; --- Integrate Results into Content ---
    (let [final-content (reduce (fn [content block]
                                  (let [block-output (get @execution-cache (:id block))      ; Get block's output
                                        code-display (wrap-code-block              ; Format code display
                                                      (:code block) "python")
                                        output-tag (str "<<output id=\""           ; Build output marker
                                                        (:id block) "\">><</output>>")]
                                    (if block-output
                                      (-> content
                                        ;; Handle both inline and separated output cases
                                          (str/replace (:full-match block)
                                                       (if (str/includes? content output-tag)
                                                         code-display             ; Code only if output tag exists
                                                         (str code-display block-output)))  ; Code + output inline
                                          (str/replace output-tag block-output))  ; Replace output tag if present
                                      (do
                                        (println "Warning: No cached output for block" (:id block))
                                        content))))
                                content                                          ; Initial content
                                blocks)]                                         ; Process all blocks
      (save-cache input-file @execution-cache blocks)              ; Persist final cache state
      final-content)))                                   ; Return processed content

;;; ===================================================================
;;; === Frontmatter Processing ===
;;; ===================================================================

;; Extracts and parses YAML-style frontmatter metadata from markdown documents
(defn parse-frontmatter
  "Parses YAML frontmatter from the beginning of markdown content.
   Args:
     content - String containing markdown with optional frontmatter
              Frontmatter must start and end with '---' on separate lines
   Returns:
     Map containing:
       :metadata - Parsed frontmatter key-value pairs, empty map if none found
       :content - Remaining markdown content after frontmatter"
  [content]
  (println "\n=== Parsing Frontmatter ===")
  (if (str/starts-with? content "---")                   ; Check if content starts with frontmatter marker
    (let [parts (str/split content #"---\s*\n" 3)        ; Split on frontmatter delimiters
          frontmatter (second parts)                     ; Extract frontmatter section
          body (nth parts 2)                             ; Get remaining content
          metadata (->> (str/split-lines frontmatter)    ; Process frontmatter lines:
                        (remove str/blank?)              ; Remove empty lines
                        (map #(str/split % #":\s+" 2))   ; Split each line into key-value pair
                        (into {}))]                      ; Convert pairs to map
      (println "Found metadata:" metadata)
      {:metadata metadata                                ; Return structured result
       :content body})
    (do
      (println "No frontmatter found")
      {:metadata {}                                      ; Return empty metadata if no frontmatter
       :content content})))

;;; ===================================================================
;;; === Template Processing ===
;;; ===================================================================

;; Replaces template placeholders with values from metadata and content
(defn replace-placeholders
  "Substitutes template variables with their corresponding values.
   Args:
     template - HTML template string containing {{PLACEHOLDER}} variables
     metadata - Map of frontmatter metadata values for title, date, etc
     content - Main content string to insert into template
   Returns:
     Completed HTML string with all placeholders replaced"
  [template metadata content]
  (println "\n=== Replacing Placeholders ===")
  (println "Available metadata keys:" (keys metadata))
  (-> template
      (str/replace "{{TITLE}}" (get metadata "title" ""))               ; Replace title or empty string
      (str/replace "{{DATE}}" (get metadata "date" ""))                 ; Replace date or empty string
      (str/replace "{{READING_TIME}}" (get metadata "reading_time" "")) ; Replace reading time
      (str/replace "{{CONTENT}}" content)))                             ; Insert main content

;;; ===================================================================
;;; === SSH Configuration Parsing ===
;;; ===================================================================

;; Parses command line arguments related to SSH remote execution
(defn parse-ssh-args
  "Extracts SSH connection parameters from command line arguments.
   Args:
     arg-pairs - Sequence of paired command line arguments (flags and values)
   Returns:
     Map containing SSH connection parameters if any are specified:
       :host - Remote server hostname/IP
       :port - SSH port number (defaults to 22)
       :user - SSH username
       :key  - Path to SSH private key file (optional)
     Returns nil if no SSH parameters are specified
   Side effects:
     Prints configuration summary if SSH parameters are found"
  [arg-pairs]
  (let [;; Extract SSH hostname from --ssh-host or -sh flag
        host (some #(when (or (= "--ssh-host" (first %))
                              (= "-sh" (first %)))
                      (second %))
                   arg-pairs)

        ;; Extract SSH username from --ssh-user or -su flag
        user (some #(when (or (= "--ssh-user" (first %))
                              (= "-su" (first %)))
                      (second %))
                   arg-pairs)

        ;; Extract SSH port from --ssh-port or -sp flag, default to 22
        port (or (some #(when (or (= "--ssh-port" (first %))
                                  (= "-sp" (first %)))
                          (second %))
                       arg-pairs)
                 "22")

        ;; Extract SSH key path from --ssh-key or -sk flag
        key (some #(when (or (= "--ssh-key" (first %))
                             (= "-sk" (first %)))
                     (second %))
                  arg-pairs)]

    ;; Only return SSH config if both host and user are specified
    (when (and host user)
      (let [config {:host host
                    :user user
                    :port port
                    :key key}]
        (println "\n=== SSH Configuration ===")
        (println "Host:" host)
        (println "User:" user)
        (println "Port:" port)
        (when key (println "Key:" key))
        config))))

;;; ===================================================================
;;; === Entry Point ===
;;; ===================================================================

;; Processes and validates command line arguments, displaying help if needed
(defn parse-cli-args
  "Parses command line arguments into a configuration map for site generation.
   Args:
     args - Sequence of command line arguments:
            First arg must be input file path
            Remaining args can be option flags with values:
              -c, --compute <id>    : Compute specific block
              -re, --recompute      : Force recomputation of all blocks
              -o, --output <file>   : Override default output path
              -sh, --ssh-host <host>: Remote SSH hostname/IP
              -su, --ssh-user <user>: Remote SSH username
              -sp, --ssh-port <port>: Remote SSH port (default: 22)
              -sk, --ssh-key <path> : Remote SSH private key path
   Returns:
     Map containing parsed configuration:
       :input-file  - Path to input markdown file
       :output-file - Path to output HTML file
       :compute-id  - Optional ID of block to compute
       :recompute?  - Flag to force recomputation
       :ssh-config  - Map containing SSH connection parameters or nil
   Side effects:
     Prints help text and exits if args are invalid"
  [args]
  (when (empty? args)
    (println "Error: Input file required")
    (println "Usage: ./site-gen.bb <input-file> [options]")
    (println "Options:")
    (println "  -c, --compute <id>     Compute specific block ID")
    (println "  -re, --recompute       Force recomputation of all blocks")
    (println "  -o, --output <file>    Specify output file (default: index.html)")
    (println "  -sh, --ssh-host <host> Remote SSH server hostname/IP for execution")
    (println "  -su, --ssh-user <user> Remote SSH username for execution")
    (println "  -sp, --ssh-port <port> Remote SSH port (default: 22)")
    (println "  -sk, --ssh-key <path>  Path to SSH private key for authentication")
    (System/exit 1))

  (let [;; Extract required input file from first argument
        input-file (first args)
        ;; Group remaining args into pairs, padding last pair if needed
        arg-pairs (partition 2 2 nil (rest args))

        ;; Find compute block ID from -c/--compute flag if present
        compute-id (some #(when (or (= "--compute" (first %))
                                    (= "-c" (first %)))
                            (second %))
                         arg-pairs)

        ;; Extract custom output path from -o/--output flag
        output-file (some #(when (or (= "--output" (first %))
                                     (= "-o" (first %)))
                             (second %))
                          arg-pairs)

        ;; Check for recompute flag presence
        recompute? (some #(or (= "--recompute" (first %))
                              (= "-re" (first %)))
                         arg-pairs)

        ;; Parse SSH configuration parameters
        ssh-config-map (parse-ssh-args arg-pairs)]

    ;; Return parsed configuration map
    {:input-file input-file
     :output-file (or output-file output-file)  ; Use provided output path or default
     :compute-id compute-id                     ; Block ID to compute (if any)
     :recompute? recompute?                     ; Whether to force recomputation
     :ssh-config ssh-config-map}))              ; SSH configuration if remote execution requested

;;; ===================================================================
;;; === Main Entry Point and Site Generation ===
;;; ===================================================================

;; Main orchestration function that coordinates the entire site generation process:
;; 1. Parses command line arguments
;; 2. Loads and processes input markdown
;; 3. Executes code blocks with caching
;; 4. Converts content through markdown/HTML pipeline
;; 5. Applies template and writes final output
(defn -main
  "Primary entry point that processes markdown into a complete HTML site.
   Args:
     & args - Command line arguments that control processing:
             - Required input file path
             - Optional flags for compute/recompute/output/ssh
   Returns:
     nil, but generates HTML output file as side effect
   Side effects:
     - Reads input markdown and template files
     - Executes Python code blocks locally or remotely
     - Writes generated HTML to output file
     - Prints status messages to console"
  [& args]
  (println "\n=== Starting Site Generation ===")

  (let [;; Parse command line args into processing configuration
        {:keys [input-file output-file compute-id recompute?]} (parse-cli-args args)
        config-map (parse-cli-args args)
        ssh-config-value (:ssh-config config-map)]

    ;; Set up SSH configuration if specified
    (when ssh-config-value
      ;; Update the atom with the ssh config
      (reset! ssh-config ssh-config-value)

      ;; Test the SSH connection and set up remote environment
      (when (test-ssh-connection ssh-config-value)
        (setup-remote-environment ssh-config-value)))

    (let [;; Load and validate input markdown file
          raw-md (try
                   (slurp input-file)
                   (catch java.io.FileNotFoundException e
                     (println "Error: Input file" input-file "not found")
                     (System/exit 1)))

          ;; Extract frontmatter metadata and main content
          {:keys [metadata content]} (parse-frontmatter raw-md)

          ;; Process content through transformation pipeline:
          executed-content (execute-blocks input-file content compute-id recompute?)  ; 1. Execute code blocks
          processed-content (-> executed-content
                                process-code-blocks                                   ; 2. Format code display
                                process-sidenotes                                     ; 3. Handle sidenotes
                                markdown->html)                                       ; 4. Convert markdown to HTML

          ;; Load template and integrate content
          template (slurp template-file)
          final-html (replace-placeholders template metadata processed-content)]

      ;; Write completed HTML output
      (println "\n=== Writing Output ===")
      (spit output-file final-html)
      (println "Generated" output-file)

      ;; Clean up remote environment if it was used
      (when @ssh-config
        (cleanup-remote-environment @ssh-config)))))

;;; ===================================================================
;;; === Script Initialization ===
;;; ===================================================================

;; Execute main function if running as a script rather than being required as a module
(when (= *file* (System/getProperty "babashka.file"))   ; Check if file is being executed directly
  (apply -main *command-line-args*))                    ; Pass command line args to main function

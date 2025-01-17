#!/usr/bin/env bb

;;; === Local Development Server Documentation ===
;;; This script provides a simple HTTP server for local development:
;;; - Serves static HTML files
;;; - Configurable port and host
;;; - Interactive shutdown via keyboard
;;; - Command-line argument support

(require '[org.httpkit.server :as server]
         '[clojure.java.io :as io])

;;; === Configuration ===
(def default-config
  "Default server configuration parameters"
  {:port 3002
   :host "localhost"
   :input-file "index.html"})

;;; === Command Line Processing ===

;; Processes command line arguments into a configuration map
(defn parse-args
  "Parses command line arguments into server configuration.
   Args:
     args - Sequence of command line argument strings
   Returns:
     Map containing:
       :input-file - Path to HTML file to serve
       :port - Server port number
       :host - Host address to bind to"
  [args]
  (let [arg-pairs (partition 2 args)                    ; Group args into pairs
        ;; Extract input file from -i/--input flag or use default
        input-file (or (second (first (filter #(or (= "-i" (first %))
                                                  (= "--input" (first %)))
                                            arg-pairs)))
                      (:input-file default-config))
        ;; Extract port from -p/--port flag or use default
        port (or (some-> (first (filter #(or (= "-p" (first %))
                                           (= "--port" (first %)))
                                     arg-pairs))
                        second
                        (Integer/parseInt))
                 (:port default-config))
        ;; Extract host from --host flag or use default
        host (or (second (first (filter #(= "--host" (first %))
                                      arg-pairs)))
                (:host default-config))]
    {:input-file input-file
     :port port
     :host host}))

;;; === Server Handlers ===

;; Creates HTTP request handler function for serving static file
(defn handler
  "Creates a ring handler function that serves a static HTML file.
   Args:
     input-file - Path to HTML file to serve
   Returns:
     Handler function that takes a request map and returns response map"
  [input-file]
  (fn [req]
    (try
      {:status 200
       :headers {"Content-Type" "text/html"}
       :body (slurp input-file)}                        ; Read and serve file content
      (catch java.io.FileNotFoundException e            ; Handle missing file
        {:status 404
         :headers {"Content-Type" "text/plain"}
         :body (str "File not found: " input-file)}))))

;;; === Interactive Control ===

;; Monitors keyboard input for server shutdown command
(defn wait-for-key
  "Blocks and waits for 'c' key press to initiate server shutdown.
   Args:
     None
   Returns:
     Never returns normally (exits process on key press)"
  []
  (let [reader (java.io.BufferedReader. *in*)]
    (while true                                         ; Continue until 'c' pressed
      (when (= (char (.read reader)) \c)
        (println "\nStopping server...")
        (System/exit 0)))))

;;; === Help Documentation ===

;; Prints usage instructions and command line options
(defn print-help
  "Displays command-line usage instructions and examples.
   Args:
     None
   Returns:
     nil"
  []
  (println "Usage: ./serve.bb [options]")
  (println "\nOptions:")
  (println "  -i, --input <file>    HTML file to serve (default: index.html)")
  (println "  -p, --port <number>   Port to run server on (default: 3002)")
  (println "  --host <ip>           Host IP address (default: localhost)")
  (println "  -h, --help            Show this help message")
  (println "\nExamples:")
  (println "  ./serve.bb")
  (println "  ./serve.bb -i custom.html --port 8080")
  (println "  ./serve.bb --host 192.168.1.100 --port 3000"))

;;; === Main Entry Point ===

;; Primary entry point that initializes and runs the server
(defn -main
  "Starts HTTP server with specified configuration.
   Args:
     & args - Command line arguments for server configuration
   Returns:
     nil, but keeps server running until interrupted"
  [& args]
  ;; Show help if requested
  (when (and (seq args)
             (or (= (first args) "-h")
                 (= (first args) "--help")))
    (print-help)
    (System/exit 0))

  ;; Start server with parsed configuration
  (let [{:keys [input-file port host]} (parse-args args)]
    (println (format "Starting server at http://%s:%d" host port))
    (println (format "Serving file: %s" input-file))
    (println "Press '⌃ + C' to stop the server")
    (try
      (let [server (server/run-server (handler input-file)
                                     {:port port
                                      :ip host})]
        (.start (Thread. wait-for-key))                 ; Start keyboard monitor thread
        @(promise))                                     ; Keep main thread alive
      (catch java.net.BindException e                   ; Handle port binding errors
        (println (format "Error: Could not bind to %s:%d - port may be in use" host port))
        (System/exit 1))
      (catch Exception e                                ; Handle other errors
        (println "Error starting server:" (.getMessage e))
        (System/exit 1)))))

;;; === Script Initialization ===

;; Execute main function if running as script
(when (= *file* (System/getProperty "babashka.file"))
  (apply -main *command-line-args*))

#!/usr/bin/env bb

(require '[clojure.java.shell :refer [sh]]
         '[clojure.string :as str]
         '[babashka.fs :as fs])

(def repl-file "/tmp/python_repl.sock")
(def repl-pid-file "/tmp/python_repl.pid")

(defn ensure-repl-dir []
  (fs/create-dirs (fs/parent repl-file)))

(defn start-repl []
  (ensure-repl-dir)
  (spit "/tmp/repl_server.py" "
import socket
import sys
import os
import time
from io import StringIO

def create_repl_server(sock_path):
    # Clean up old socket if it exists
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    # Change socket permissions to allow other users to connect
    os.chmod(sock_path, 0o777)

    globals_dict = {}
    locals_dict = {}

    print(f'REPL server started and listening on: {sock_path}')
    sys.stdout.flush()

    while True:
        conn, addr = server.accept()
        try:
            data = conn.recv(4096).decode()
            output_buffer = StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_buffer

            try:
                exec(data, globals_dict, locals_dict)
                output = output_buffer.getvalue()
                conn.send(output.encode() or b'No output')
            except Exception as e:
                conn.send(str(e).encode())
            finally:
                sys.stdout = original_stdout
        finally:
            conn.close()

if __name__ == '__main__':
    create_repl_server('" repl-file "')")

  (let [proc (-> (Runtime/getRuntime)
                 (.exec (into-array ["python3" "/tmp/repl_server.py"])))]
    (spit repl-pid-file (.pid proc))
    (println "Starting REPL server. PID:" (.pid proc))

    ; Wait for server to start and create socket
    (Thread/sleep 1000)

    (if (fs/exists? repl-file)
      (println "REPL server successfully started and socket created.")
      (println "Error: REPL server failed to create socket file."))))

(defn send-to-repl [code]
  (if-not (fs/exists? repl-file)
    (println "Error: REPL server not running. Start it with './py_repl.clj start'")
    (do
      (spit "/tmp/repl_client.py" (str "
import socket
import sys

code = '''" code "'''

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    sock.connect('" repl-file "')
    sock.send(code.encode())
    response = sock.recv(4096).decode()
    print(response, end='')
except Exception as e:
    print(f'Error: {str(e)}', file=sys.stderr)
finally:
    sock.close()"))

      (let [result (sh "python3" "/tmp/repl_client.py")]
        (when (not-empty (:out result))
          (print (:out result)))
        (when (not-empty (:err result))
          (println "Error:" (:err result)))))))

(defn stop-repl []
  (when (fs/exists? repl-pid-file)
    (try
      (let [pid (slurp repl-pid-file)]
        (sh "kill" "-9" pid)
        (fs/delete repl-pid-file))
      (catch Exception e
        (println "Error stopping REPL process:" (str e)))))

  (when (fs/exists? repl-file)
    (try
      (fs/delete repl-file)
      (catch Exception e
        (println "Error removing socket file:" (str e)))))

  (println "REPL server stopped."))

(defn -main [& args]
  (if (empty? args)
    (println "Usage: py_repl.clj [start|stop|send <code>]")
    (case (first args)
      "start" (start-repl)
      "stop" (stop-repl)
      "send" (if-let [code (second args)]
               (send-to-repl code)
               (println "Error: No code provided for send command"))
      (println "Unknown command. Use start, stop, or send."))))

(when (= *file* (System/getProperty "babashka.file"))
  (apply -main *command-line-args*))

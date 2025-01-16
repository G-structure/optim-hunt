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
  (spit "/tmp/repl_server.py"
    (str "
import socket
import sys
import os
import time
from io import StringIO
import select

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
        readable, _, _ = select.select([server], [], [], 1.0)
        if not readable:
            continue

        conn, addr = server.accept()
        conn.settimeout(5.0)  # Set timeout for operations
        try:
            data = conn.recv(4096).decode()
            if not data:
                continue

            output_buffer = StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_buffer

            try:
                exec(data, globals_dict, locals_dict)
                output = output_buffer.getvalue()
                conn.send((output or 'No output').encode())
            except Exception as e:
                conn.send(str(e).encode())
            finally:
                sys.stdout = original_stdout
        except socket.timeout:
            conn.send(b'Operation timed out')
        except Exception as e:
            conn.send(str(e).encode())
        finally:
            conn.close()

if __name__ == '__main__':
    create_repl_server('" repl-file "')"))

  (let [process-builder (ProcessBuilder. ["python3" "/tmp/repl_server.py"])
        _ (.redirectErrorStream process-builder true)
        proc (.start process-builder)]

    (spit repl-pid-file (.pid proc))
    (println "Starting REPL server. PID:" (.pid proc))

    ; Wait for server to start and create socket
    (Thread/sleep 2000)

    (if (fs/exists? repl-file)
      (println "REPL server successfully started and socket created.")
      (do
        (println "Error: REPL server failed to create socket file.")
        ; Print any error output
        (with-open [reader (java.io.BufferedReader.
                           (java.io.InputStreamReader.
                            (.getInputStream proc)))]
          (loop []
            (when-let [line (.readLine reader)]
              (println line)
              (recur))))))))

(defn send-to-repl [code]
  (if-not (fs/exists? repl-file)
    (println "Error: REPL server not running. Start it with './py_repl.clj start'")
    (do
      (spit "/tmp/repl_client.py"
        (str "
import socket
import sys
import select

code = '''" code "'''

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(5.0)  # Set timeout for connection

try:
    sock.connect('" repl-file "')
    sock.send(code.encode())

    # Wait for response with timeout
    ready = select.select([sock], [], [], 5.0)
    if ready[0]:
        response = sock.recv(4096).decode()
        print(response, end='')
    else:
        print('Response timeout', file=sys.stderr)
except socket.timeout:
    print('Connection/operation timed out', file=sys.stderr)
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
        (sh "kill" pid)
        (Thread/sleep 1000)
        ; Force kill if process is still running
        (try (sh "kill" "-9" pid) (catch Exception _))
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

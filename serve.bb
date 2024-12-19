#!/usr/bin/env bb

(require '[org.httpkit.server :as server])
(require '[clojure.java.io :as io])

(def port 3002)

(defn handler [req]
  {:status 200
   :headers {"Content-Type" "text/html"}
   :body (slurp "index.html")})

(defn wait-for-key []
  (let [reader (java.io.BufferedReader. *in*)]
    (while true
      (when (= (char (.read reader)) \c)
        (println "\nStopping server...")
        (System/exit 0)))))

(defn -main []
  (println (str "Starting server at http://localhost:" port))
  (println "Press 'c' to stop the server")
  (let [server (server/run-server handler {:port port})]
    (.start (Thread. wait-for-key))
    @(promise)))

(-main)
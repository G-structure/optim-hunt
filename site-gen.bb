#!/usr/bin/env bb

(require '[clojure.string :as str]
         '[clojure.java.shell :as shell]
         '[clojure.java.io :as io]
         '[cheshire.core :as json])

;; --- Config ---
(def template-file "template.html")
(def content-file "content.md")
(def output-file "index.html")
(def cache-file ".code-outputs.json")

;; --- Cache Management ---
(defn load-cache []
  (println "\n=== Loading Cache ===")
  (let [cache-path cache-file]
    (when (.exists (io/file cache-path))
      (println "Cache file exists at:" cache-path)
      (try
        (let [cache-content (slurp cache-path)
              _ (println "Cache content length:" (count cache-content))
              ;; Read cache as plain string and parse manually
              cache (if (str/starts-with? cache-content "{")
                     (let [content (-> cache-content
                                     (subs 1 (dec (count cache-content))) ; Remove outer {}
                                     (str/split #"\,(?=\s*\"[^\"]+\":)"))] ; Split on commas between key-value pairs
                       (into {}
                             (for [pair content]
                               (let [[k v] (str/split pair #":" 2)
                                     key (str/replace (str/trim k) #"\"" "")
                                     val (str/trim v)]
                                 [key val]))))
                     {})]
          (println "Successfully parsed cache with" (count cache) "entries")
          (println "Cache keys:" (keys cache))
          cache)
        (catch Exception e
          (println "Error reading cache:" (str e))
          {})))))

(defn save-cache [cache]
  (println "\n=== Saving Cache ===")
  (println "Saving" (count cache) "outputs to cache")
  (let [cache-str (str "{"
                      (str/join ","
                               (for [[k v] cache]
                                 (format "\"%s\":%s" k v)))
                      "}")]
    (spit cache-file cache-str)))

;; --- Code formatting ---
(defn wrap-code-block [code lang]
  (format "<div class=\"code-block\"><pre class=\"line-numbers\"><code class=\"language-%s\">%s</code></pre></div>"
          lang
          (-> code
              (str/replace "<" "&lt;")
              (str/replace ">" "&gt;"))))

(defn process-code-blocks [content]
  (println "\n=== Processing Code Blocks ===")
  (let [code-block-pattern #"(?ms)```(\w+)\n(.*?)\n```"]
    (str/replace content code-block-pattern
                (fn [[_ lang code]]
                  (if (str/includes? content "<<execute")
                    (str "```" lang "\n" code "\n```")
                    (wrap-code-block code lang))))))

;; --- Code Execution and Processing ---
(defn is-html? [s]
  (or (str/starts-with? (str/trim s) "<")
      (str/starts-with? (str/trim s) "<!DOCTYPE")))

(defn run-python [code]
  (println "\n=== Executing Python Code ===")
  (println "Code to execute:\n" code)
  (try
    (let [res (shell/sh "python3" "-c" code)]
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
      (println "Exception running Python:" (str e))
      (str "<pre class=\"code-error\">Error: " (str e) "</pre>"))))

(defn markdown->html [markdown]
  (println "\n=== Converting Markdown to HTML ===")
  (let [res (shell/sh "pandoc"
                      "--from=markdown+tex_math_dollars+tex_math_single_backslash"
                      "--to=html"
                      "--mathjax"
                      "--wrap=none"
                      :in markdown)]
    (if (zero? (:exit res))
      (do
        (println "Conversion successful")
        (:out res))
      (do (println "Pandoc error:" (:err res))
          markdown))))

(defn process-sidenotes [content]
  (println "\n=== Processing Sidenotes ===")
  (let [sidenote-pattern #"(?s)\^\^\^(.*?)\^\^\^"
        matches (re-seq sidenote-pattern content)]
    (println "Found" (count matches) "sidenotes")
    (let [process-single-sidenote (fn [[_ note-content]]
                                   (let [processed-content (markdown->html note-content)]
                                     (format "<div class=\"sidenote\">%s</div>"
                                           (str/trim processed-content))))]
      (str/replace content sidenote-pattern process-single-sidenote))))

;; --- Block Parsing ---
(defn find-code-blocks [content]
  (println "\n=== Finding Code Blocks ===")
  (let [block-pattern #"(?ms)<<execute\s+id=\"(\d+)\"\s+output=\"([^\"]+)\">>\s*```python\n(.*?)\n```\s*<<\/execute>>"
        matches (re-seq block-pattern content)]
    (println "Found" (count matches) "code blocks")
    (let [blocks (map (fn [[full-match id output-type code]]
                       {:id id
                        :output-type output-type
                        :code (str/trim code)
                        :full-match full-match})
                     matches)]
      (let [ids (map :id blocks)
            duplicates (keep (fn [[id freq]] (when (> freq 1) id))
                           (frequencies ids))]
        (when (seq duplicates)
          (throw (Exception. (str "Duplicate block IDs found: " (str/join ", " duplicates)))))
        blocks))))

;; --- Output Processing ---
(defn process-output [output output-type]
  (println "\n=== Processing Output ===")
  (println "Processing output of type:" output-type)
  (case output-type
    "raw" (do (println "Using raw output") output)
    "pandoc" (do
               (println "Converting output through pandoc")
               (let [res (shell/sh "pandoc" "--from=markdown" "--to=html"
                                 :in output)]
                 (if (zero? (:exit res))
                   (:out res)
                   (do (println "Pandoc error:" (:err res))
                       output))))
    (throw (Exception. (str "Invalid output type: " output-type)))))

;; --- Block Execution ---
(defn execute-blocks [content compute-id]
  (println "\n=== Executing Blocks ===")
  (when compute-id
    (println "Computing only block with ID:" compute-id))

  (let [cache (atom (load-cache))
        blocks (find-code-blocks content)]

    (doseq [block blocks]
      (println "\nProcessing block" (:id block))
      (when (or (nil? compute-id) (= (:id block) compute-id))
        (let [output (run-python (:code block))
              processed (process-output output (:output-type block))]
          (swap! cache assoc (:id block) processed)
          (println "Updated cache for block" (:id block)))))

    (let [final-content (reduce (fn [content block]
                                (let [block-output (get @cache (:id block))
                                      code-display (wrap-code-block (:code block) "python")
                                      output-tag (str "<<output id=\"" (:id block) "\">><</output>>")]
                                  (if block-output
                                    (-> content
                                        (str/replace (:full-match block)
                                                   (if (str/includes? content output-tag)
                                                     code-display
                                                     (str code-display block-output)))
                                        (str/replace output-tag block-output))
                                    (do
                                      (println "Warning: No cached output for block" (:id block))
                                      content))))
                              content
                              blocks)]
      (save-cache @cache)
      final-content)))

;; --- Frontmatter Processing ---
(defn parse-frontmatter [content]
  (println "\n=== Parsing Frontmatter ===")
  (if (str/starts-with? content "---")
    (let [parts (str/split content #"---\s*\n" 3)
          frontmatter (second parts)
          body (nth parts 2)
          metadata (->> (str/split-lines frontmatter)
                       (remove str/blank?)
                       (map #(str/split % #":\s+" 2))
                       (into {}))]
      (println "Found metadata:" metadata)
      {:metadata metadata
       :content body})
    (do
      (println "No frontmatter found")
      {:metadata {}
       :content content})))

(defn replace-placeholders [template metadata content]
  (println "\n=== Replacing Placeholders ===")
  (println "Available metadata keys:" (keys metadata))
  (-> template
      (str/replace "{{TITLE}}" (get metadata "title" ""))
      (str/replace "{{DATE}}" (get metadata "date" ""))
      (str/replace "{{READING_TIME}}" (get metadata "reading_time" ""))
      (str/replace "{{CONTENT}}" content)))

;; --- Main ---
(defn -main [& args]
  (println "\n=== Starting Site Generation ===")
  (let [compute-id (second (first (filter #(= "--compute" (first %)) (partition 2 args))))
        _ (println "Reading content file:" content-file)
        raw-md (slurp content-file)
        _ (println "Raw content length:" (count raw-md))
        {:keys [metadata content]} (parse-frontmatter raw-md)
        _ (println "Content after frontmatter length:" (count content))
        executed-content (execute-blocks content compute-id)
        processed-content (-> executed-content
                            process-code-blocks
                            process-sidenotes
                            markdown->html)
        template (slurp template-file)
        final-html (replace-placeholders template metadata processed-content)]
    (println "\n=== Writing Output ===")
    (spit output-file final-html)
    (println "Generated" output-file)))

(when (= *file* (System/getProperty "babashka.file"))
  (apply -main *command-line-args*))

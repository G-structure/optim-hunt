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
  (try
    (let [cache (json/parse-string (slurp cache-file))]
      (println "Loaded" (count cache) "cached outputs")
      cache)
    (catch Exception _
      (println "No cache file found or error reading cache. Starting with empty cache.")
      {})))

(defn save-cache [cache]
  (println "\n=== Saving Cache ===")
  (println "Saving" (count cache) "outputs to cache")
  (spit cache-file (json/generate-string cache)))

;; --- Code formating
(defn wrap-code-block [code lang]
  (format "<div class=\"code-block\"><pre class=\"line-numbers\"><code class=\"language-%s\">%s</code></pre></div>"
          lang
          (-> code
              (str/replace "<" "&lt;")
              (str/replace ">" "&gt;"))))

(defn process-code-blocks [content]
  (println "\n=== Processing Code Blocks ===")
  ;; Handle regular markdown code blocks
  (let [code-block-pattern #"(?ms)```(\w+)\n(.*?)\n```"]
    (str/replace content code-block-pattern
                (fn [[_ lang code]]
                  (when-not (str/includes? content (format "<<execute") )
                    (wrap-code-block code lang))))))

;; --- Code Execution and Processing ---
(defn is-html? [s]
  (or (str/starts-with? (str/trim s) "<")
      (str/starts-with? (str/trim s) "<!DOCTYPE")))

(defn run-python [code]
  (println "\n=== Executing Python Code ===")
  (println "Code to execute:\n" code)
  (let [res (shell/sh "python3" "-c" code)]
    (if (zero? (:exit res))
      (let [output (:out res)]
        (println "Execution successful. Output length:" (count output))
        (if (is-html? output)
          output
          (str "<pre class=\"code-output\">" output "</pre>")))
      (do
        (println "Error running python code:" (:err res))
        (str "<pre class=\"code-error\">" (:err res) "</pre>")))))

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
(defn parse-execute-tag [tag]
  (let [id-match (re-find #"id=\"(\d+)\"" tag)
        output-match (re-find #"output=\"(raw|pandoc)\"" tag)]
    (when (not id-match)
      (throw (Exception. "Execute tag missing required id attribute")))
    {:id (second id-match)
     :output-type (or (second output-match) "raw")}))

(defn find-code-blocks [content]
  (println "\n=== Finding Code Blocks ===")
  (let [block-pattern #"(?ms)<<execute\s+id=\"(\d+)\"\s+output=\"([^\"]+)\">>\s*```python\n(.*?)\n```\s*<<\/execute>>"
        matches (re-seq block-pattern content)]
    (println "Found" (count matches) "code blocks")
    (doseq [[_ id output-type code] matches]
      (println "\nBlock ID:" id)
      (println "Output type:" output-type)
      (println "Code preview:" (subs code 0 (min 50 (count code))) "..."))

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

    ;; Process each block
    (doseq [block blocks]
      (println "\nProcessing block" (:id block))
      (if (= (:id block) compute-id)
        ;; If this is the block we want to compute, execute it regardless of cache
        (do
          (println "Forcing execution of block" (:id block))
          (let [output (run-python (:code block))
                processed (process-output output (:output-type block))]
            (swap! cache assoc (:id block) processed)
            (println "Updated cache for block" (:id block))))
        ;; For all other blocks, use cache if available
        (if-let [cached-output (get @cache (:id block))]
          (println "Using cached output for block" (:id block))
          (do
            ;; Only execute if no cache available
            (println "No cache found for block" (:id block) ". Executing...")
            (let [output (run-python (:code block))
                  processed (process-output output (:output-type block))]
              (swap! cache assoc (:id block) processed)
              (println "Cached new output for block" (:id block)))))))

    ;; Replace blocks in content with both code display and output
    (let [final-content (reduce (fn [content block]
                                (let [block-output (get @cache (:id block))
                                      code-display (wrap-code-block (:code block) "python")
                                      output-tag (str "<<output id=\"" (:id block) "\">><</output>>")]
                                  (if block-output
                                    (-> content
                                        (str/replace (:full-match block)
                                                   (if (str/includes? content output-tag)
                                                     code-display ; Show code but remove execute block
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

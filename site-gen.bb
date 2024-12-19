#!/usr/bin/env bb

(require '[clojure.string :as str]
         '[clojure.java.shell :as shell])

;; --- Config ---
(def template-file "template.html")
(def content-file "content.md")
(def output-file "index.html")

;; Parse YAML frontmatter
(defn parse-frontmatter [content]
  (if (str/starts-with? content "---")
    (let [parts (str/split content #"---\s*\n" 3)
          frontmatter (second parts)
          body (nth parts 2)
          metadata (->> (str/split-lines frontmatter)
                       (remove str/blank?)
                       (map #(str/split % #":\s+" 2))
                       (into {}))]
      {:metadata metadata
       :content body})
    {:metadata {}
     :content content}))

(defn is-html? [s]
  (or (str/starts-with? (str/trim s) "<")
      (str/starts-with? (str/trim s) "<!DOCTYPE")))

(defn run-python [code]
  (let [res (shell/sh "python3" "-c" code)]
    (if (zero? (:exit res))
      (let [output (:out res)]
        (if (is-html? output)
          output  ; Return HTML directly without wrapping
          (str "<pre class=\"code-output\">" output "</pre>")))  ; Wrap non-HTML output
      (do
        (println "Error running python code:" (:err res))
        (str "<pre class=\"code-error\">" (:err res) "</pre>")))))

(defn replace-python-blocks [markdown]
  (let [pattern #"(?s)```([a-zA-Z0-9]+)\n(.*?)```"]
    (str/replace markdown pattern
                 (fn [[_ lang code]]
                   (if (= (str/lower-case lang) "python")
                     (str (run-python code))
                     (str "<pre class=\"code-block\"><code class=\"language-" lang "\">" 
                          (str/escape code 
                                     {\< "&lt;"
                                      \> "&gt;"
                                      \& "&amp;"})
                          "</code></pre>"))))))

;; Convert markdown to HTML using pandoc
(defn markdown->html [markdown]
  (let [res (shell/sh "pandoc"
                      "--from=markdown+tex_math_dollars+tex_math_single_backslash"
                      "--to=html"
                      "--mathjax"
                      "--wrap=none"
                      :in markdown)]
    (if (zero? (:exit res))
      (:out res)
      (do (println "Pandoc error:" (:err res))
          markdown))))

;; Process sidenotes: Convert ^^^ wrapped content to HTML sidenotes
(defn process-sidenotes [content]
  (let [sidenote-pattern #"(?s)\^\^\^(.*?)\^\^\^"
        process-single-sidenote (fn [[_ note-content]]
                                 (let [processed-content (markdown->html note-content)]
                                   (format "<div class=\"sidenote\">%s</div>"
                                         (str/trim processed-content))))]
    (str/replace content sidenote-pattern process-single-sidenote)))

;; Replace all placeholders in template
(defn replace-placeholders [template metadata content]
  (-> template
      (str/replace "{{TITLE}}" (get metadata "title" ""))
      (str/replace "{{DATE}}" (get metadata "date" ""))
      (str/replace "{{READING_TIME}}" (get metadata "reading_time" ""))
      (str/replace "{{CONTENT}}" content)))

;; MAIN
(defn -main []
  (let [raw-md (slurp content-file)
        {:keys [metadata content]} (parse-frontmatter raw-md)
        processed-md-with-sidenotes (process-sidenotes content)
        processed-md-with-python (replace-python-blocks processed-md-with-sidenotes)
        content-html (markdown->html processed-md-with-python)
        template (slurp template-file)
        final-html (replace-placeholders template metadata content-html)]
    (spit output-file final-html)
    (println (str "Generated " output-file))))

(-main)

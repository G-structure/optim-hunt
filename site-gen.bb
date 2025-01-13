#!/usr/bin/env bb

;;; === Site Generator Documentation ===
;;; This script generates HTML content from markdown files with special features:
;;; - Executes Python code blocks and captures their output
;;; - Converts markdown to HTML using pandoc
;;; - Handles frontmatter metadata
;;; - Processes sidenotes
;;; - Caches code execution results

(require '[clojure.string :as str]
         '[clojure.java.shell :as shell]
         '[clojure.java.io :as io]
         '[cheshire.core :as json])

;;; === Configuration ===
(def template-file "template.html")
(def output-file "index.html")
(def cache-file ".code-outputs.json")
(def cache-dir ".cache")

;;; === Cache Management Functions ===

;; Ensures cache directory exists and creates it if needed
(defn ensure-cache-dir!
  "Creates cache directory if it doesn't exist.
   Args:
     None
   Returns:
     true if directory exists or was created successfully"
  []
  (let [dir (io/file cache-dir)]
    (when-not (.exists dir)
      (println "Creating cache directory:" cache-dir)
      (.mkdir dir))
    (.exists dir)))

;; Generates cache file path for a specific block
(defn cache-file-path
  "Constructs path for block-specific cache file.
   Args:
     input-file - Source markdown file path
     block-id   - Unique identifier for code block
   Returns:
     String path to cache file"
  [input-file block-id]
  (let [base-name (-> input-file
                      io/file
                      .getName
                      (str/replace #"\.[^.]+$" ""))]    ; Remove file extension
    (str cache-dir
         "/"
         base-name
         "-block-"
         block-id
         ".edn")))

;; Loads cached output for a specific code block
(defn load-block-cache
  "Reads cached output for a specific code block.
   Args:
     input-file - Source markdown file path
     block-id   - Unique identifier for code block
   Returns:
     Cached output string if available, nil if no cache exists"
  [input-file block-id]
  (println "\n=== Loading Block Cache ===")
  (ensure-cache-dir!)
  (let [cache-path (cache-file-path input-file block-id)]
    (println "Checking cache file:" cache-path)
    (when (.exists (io/file cache-path))
      (try
        (let [cached-data (read-string (slurp cache-path))]  ; Read EDN format
          (println "Found cached output for block:" block-id)
          (:output cached-data))                            ; Extract output from cache data
        (catch Exception e
          (println "Error reading cache for block" block-id ":" (str e))
          nil)))))

;; Loads all cached outputs for an input file
(defn load-cache
  "Reads all cached block outputs for a given input file.
   Args:
     input-file - Source markdown file path
   Returns:
     Map of block IDs to their cached outputs"
  [input-file]
  (println "\n=== Loading Full Cache ===")
  (ensure-cache-dir!)
  (let [cache-pattern (re-pattern (str (-> input-file
                                          io/file
                                          .getName
                                          (str/replace #"\.[^.]+$" ""))
                                      "-block-\\d+\\.edn"))
        cache-files (filter #(re-matches cache-pattern (.getName %))
                           (.listFiles (io/file cache-dir)))]
    (into {}
          (for [cache-file cache-files
                :let [block-id (second (re-find #"-block-(\d+)\.edn$" (.getName cache-file)))
                      cached-data (try
                                  (read-string (slurp cache-file))
                                  (catch Exception e nil))]
                :when cached-data]
            [block-id (:output cached-data)]))))

;; Saves execution output for a specific block
(defn save-block-cache
  "Persists code block output to cache file.
   Args:
     input-file - Source markdown file path
     block-id   - Unique identifier for code block
     code       - Source code that was executed
     output     - Execution output to cache
   Returns:
     nil"
  [input-file block-id code output]
  (println "\n=== Saving Block Cache ===")
  (ensure-cache-dir!)
  (let [cache-path (cache-file-path input-file block-id)
        cache-data {:block-id block-id                  ; Store metadata with output
                   :code code
                   :output output
                   :timestamp (System/currentTimeMillis)}]
    (println "Saving cache for block" block-id "to:" cache-path)
    (try
      (spit cache-path (pr-str cache-data))            ; Write as EDN format
      (catch Exception e
        (println "Error saving cache for block" block-id ":" (str e))))))

;; Saves all block outputs to cache
(defn save-cache
  "Persists all block outputs to cache files.
   Args:
     input-file - Source markdown file path
     cache-map  - Map of block IDs to execution outputs
     blocks     - Sequence of code block definitions
   Returns:
     nil"
  [input-file cache-map blocks]
  (println "\n=== Saving Full Cache ===")
  (doseq [block blocks
          :let [block-id (:id block)
                output (get cache-map block-id)]
          :when output]
    (save-block-cache input-file block-id (:code block) output)))

;;; === Code Block Processing ===

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
  (let [
        ;; Define regex pattern to match code blocks:
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

;;; === Code Execution ===

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

;; Executes Python code and wraps its output in appropriate HTML formatting
(defn run-python
  "Executes Python code and formats the output as HTML.
   Args:
     code - String containing Python source code to execute
   Returns:
     HTML-formatted string containing either:
     - Execution output wrapped in <pre> tags
     - Raw HTML output if result is already HTML
     - Error message wrapped in error-styled <pre> tags"
  [code]
  (println "\n=== Executing Python Code ===")
  (println "Code to execute:\n" code)
  (try
    (let [res (shell/sh "python3" "-c" code)]           ; Execute code via Python interpreter
      (if (zero? (:exit res))                           ; Check if execution was successful
        (let [output (:out res)]
          (println "Execution successful. Output length:" (count output))
          (if (is-html? output)                         ; Check if output is already HTML
            output                                      ; Return raw HTML unchanged
            (str "<pre class=\"code-output\">" output "</pre>")))   ; Wrap plain text in pre tags
        (do
          (println "Error running python code:" (:err res))
          (str "<pre class=\"code-error\">" (:err res) "</pre>")))) ; Format error output
    (catch Exception e                                   ; Handle any execution exceptions
      (println "Exception running Python:" (str e))
      (str "<pre class=\"code-error\">Error: " (str e) "</pre>"))))

;;; === Markdown Processing ===

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

;;; === Block Parsing and Execution ===

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
                        :code (str/trim code)           ; Clean up code whitespace
                        :full-match full-match})
                     matches)]
      (let [ids (map :id blocks)                        ; Extract all block IDs
            ;; Find any IDs that appear more than once
            duplicates (keep (fn [[id freq]] (when (> freq 1) id))
                           (frequencies ids))]
        (when (seq duplicates)                          ; Throw error if duplicates found
          (throw (Exception. (str "Duplicate block IDs found: " (str/join ", " duplicates)))))
        blocks))))                                      ; Return processed blocks

;;; === Output Processing ===

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
                                 "--from=markdown"      ; Input markdown format
                                 "--to=html"            ; Output HTML format
                                 :in output)]
                 (if (zero? (:exit res))               ; Check if conversion succeeded
                   (:out res)                          ; Return converted HTML
                   (do
                     (println "Pandoc error:" (:err res))
                     output))))                        ; Return original on error
    (throw (Exception. (str "Invalid output type: " output-type)))))

;;; === Main Processing Functions ===

;; Manages the execution of Python code blocks and their output integration
(defn execute-blocks
  "Executes Python code blocks and integrates their outputs into content.
   Args:
     input-file  - Source markdown file path
     content     - String containing executable code blocks
     compute-id  - Optional ID of specific block to execute (nil executes all)
   Returns:
     Content string with executed code blocks and their outputs integrated"
  [input-file content compute-id]
  (println "\n=== Executing Blocks ===")
  (when compute-id
    (println "Computing only block with ID:" compute-id))

  (let [cache (atom (load-cache input-file))          ; Pass input-file to load-cache
        blocks (find-code-blocks content)]

    ;; Execute and cache each block's output
    (doseq [block blocks]
      (println "\nProcessing block" (:id block))
      (when (or (nil? compute-id)
                (= (:id block) compute-id))
        (let [output (run-python (:code block))
              processed (process-output output (:output-type block))]
          (save-block-cache input-file (:id block) (:code block) processed)  ; Save individual block
          (swap! cache assoc (:id block) processed))))

    ;; Integrate code and outputs into content
    (let [final-content (reduce (fn [content block]
                                (let [block-output (get @cache (:id block))        ; Get cached output
                                      code-display (wrap-code-block                ; Format code display
                                                   (:code block) "python")
                                      output-tag (str "<<output id=\""             ; Build output tag
                                                 (:id block) "\">><</output>>")]
                                  (if block-output
                                    (-> content
                                        ;; Replace full block or just code based on output tag presence
                                        (str/replace (:full-match block)
                                                   (if (str/includes? content output-tag)
                                                     code-display
                                                     (str code-display block-output)))
                                        ;; Replace output tag with actual output
                                        (str/replace output-tag block-output))
                                    (do
                                      (println "Warning: No cached output for block" (:id block))
                                      content))))
                              content                                             ; Initial content
                              blocks)]                                            ; Process all blocks
      (save-cache input-file @cache blocks)                                       ; Pass input-file to save-cache
      final-content)))                                                            ; Return processed content

;;; === Frontmatter Processing ===

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
                       (remove str/blank?)               ; Remove empty lines
                       (map #(str/split % #":\s+" 2))    ; Split each line into key-value pair
                       (into {}))]                       ; Convert pairs to map
      (println "Found metadata:" metadata)
      {:metadata metadata                                ; Return structured result
       :content body})
    (do
      (println "No frontmatter found")
      {:metadata {}                                      ; Return empty metadata if no frontmatter
       :content content})))

;;; === Template Processing ===

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

;;; === Entry Point ===

;; Primary entry point function that orchestrates the site generation process
(defn -main
  "Processes input markdown file and generates HTML output with optional code execution.
   Args:
     & args - Command line arguments:
             input-file - Required path to markdown input file
             --compute block-id - Optional ID of single code block to execute
             -o/--output path - Optional output file path (defaults to index.html)
   Returns:
     nil, but writes generated HTML to output file and prints status messages"
  [& args]
  (println "\n=== Starting Site Generation ===")
  (when (empty? args)
    (println "Error: Input file required")
    (println "Usage: ./site-gen.bb <input-file> [--compute <block-id>] [-o|--output <output-file>]")
    (System/exit 1))

  (let [input-file (first args)                         ; Get input file path from first arg
        arg-pairs (partition 2 (rest args))             ; Group remaining args into flag/value pairs
        ;; Extract compute-id for single block execution
        compute-id (second (first (filter #(= "--compute" (first %)) arg-pairs)))
        ;; Get output path or use default
        output-path (or (second (first (filter #(or (= "-o" (first %))
                                                   (= "--output" (first %)))
                                             arg-pairs)))
                       output-file)
        _ (println "Reading content file:" input-file)
        _ (println "Output will be written to:" output-path)

        ;; Load and validate input file
        raw-md (try
                (slurp input-file)                      ; Attempt to read input file
                (catch java.io.FileNotFoundException e
                  (println "Error: Input file" input-file "not found")
                  (System/exit 1)))                      ; Exit if file not found
        _ (println "Raw content length:" (count raw-md))

        ;; Extract metadata and content
        {:keys [metadata content]} (parse-frontmatter raw-md)
        _ (println "Content after frontmatter length:" (count content))

        ;; Process content through transformation pipeline:
        ;; 1. Execute Python code blocks
        executed-content (execute-blocks input-file content compute-id)
        ;; 2. Convert markdown elements to HTML
        processed-content (-> executed-content
                            process-code-blocks         ; Handle code syntax highlighting
                            process-sidenotes           ; Process special sidenote syntax
                            markdown->html)             ; Convert remaining markdown to HTML

        ;; Generate final HTML document
        template (slurp template-file)                  ; Load HTML template
        final-html (replace-placeholders template metadata processed-content)]

    (println "\n=== Writing Output ===")
    (spit output-path final-html)                       ; Write generated HTML to output file
    (println "Generated" output-path)))

;;; === Script Initialization ===

;; Execute main function if running as a script rather than being required as a module
(when (= *file* (System/getProperty "babashka.file"))   ; Check if file is being executed directly
  (apply -main *command-line-args*))                    ; Pass command line args to main function

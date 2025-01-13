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

;;; === Cache Management Functions ===

;; Loads and parses the cache file containing previous code execution results
(defn load-cache
  "Reads and parses the JSON-like cache file containing code execution outputs.
   Args:
     None
   Returns:
     Map of cached outputs where keys are block IDs and values are execution results.
     Returns empty map if cache doesn't exist or is invalid."
  []
  (println "\n=== Loading Cache ===")
  (let [cache-path cache-file]                           ; Get configured cache file path
    (when (.exists (io/file cache-path))                 ; Only proceed if file exists
      (println "Cache file exists at:" cache-path)
      (try
        (let [cache-content (slurp cache-path)           ; Read entire cache file
              _ (println "Cache content length:" (count cache-content))
              ;; Parse cache content into map:
              ;; 1. Validate JSON-like structure
              ;; 2. Parse key-value pairs manually
              cache (if (str/starts-with? cache-content "{")
                     (let [content (-> cache-content
                                     (subs 1 (dec (count cache-content)))    ; Remove outer braces
                                     (str/split #"\,(?=\s*\"[^\"]+\":\)"))]  ; Split on commas before keys
                       (into {}
                             (for [pair content]
                               (let [[k v] (str/split pair #":" 2)           ; Split into key & value
                                     key (str/replace (str/trim k) #"\"" "") ; Remove quotes from key
                                     val (str/trim v)]                       ; Clean value string
                                 [key val]))))                               ; Create map entry
                     {})]                                                    ; Return empty map if invalid
          (println "Successfully parsed cache with" (count cache) "entries")
          (println "Cache keys:" (keys cache))
          cache)                                         ; Return parsed cache map
        (catch Exception e                               ; Handle any parsing errors
          (println "Error reading cache:" (str e))
          {})))))

;; Persists the code execution cache to disk storage in a JSON-like format
(defn save-cache
  "Saves the code execution cache to a file in JSON-like format.
   Args:
     cache - Map of code block IDs to their execution outputs
   Returns:
     nil"
  [cache]
  (println "\n=== Saving Cache ===")
  (println "Saving" (count cache) "outputs to cache")
  (let [cache-str (str "{"                               ; Start with opening brace
                   (str/join ","                         ; Join entries with commas
                            (for [[k v] cache]           ; Transform each cache entry
                              (format "\"%s\":%s"        ; Format as "key":value
                                     k v)))
                   "}")]                                 ; Close with ending brace
    (spit cache-file cache-str)))                        ; Write cache to disk

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
     content - String containing executable code blocks
     compute-id - Optional ID of specific block to execute (nil executes all)
   Returns:
     Content string with executed code blocks and their outputs integrated"
  [content compute-id]
  (println "\n=== Executing Blocks ===")
  (when compute-id
    (println "Computing only block with ID:" compute-id))

  (let [cache (atom (load-cache))                       ; Load existing cache into atom
        blocks (find-code-blocks content)]              ; Parse code blocks from content

    ;; Execute and cache each block's output
    (doseq [block blocks]
      (println "\nProcessing block" (:id block))
      (when (or (nil? compute-id)                       ; Execute if no specific ID requested
                (= (:id block) compute-id))             ; Or if this block matches requested ID
        (let [output (run-python (:code block))         ; Execute the Python code
              processed (process-output output          ; Process the execution output
                                      (:output-type block))]
          (swap! cache assoc (:id block) processed)     ; Cache the processed output
          (println "Updated cache for block" (:id block)))))

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
                                      content))))                                 ; Return unchanged if no output
                              content                                             ; Initial content
                              blocks)]                                            ; Process all blocks
      (save-cache @cache)                               ; Persist final cache state
      final-content)))                                  ; Return processed content

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
        executed-content (execute-blocks content compute-id)
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

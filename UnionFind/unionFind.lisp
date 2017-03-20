;
; Implementation of the UnionFind data structure described here:
; 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
;
; @2017 Florin Tulba (florintulba@yahoo.com)
;


; N, ids, ancestors and ranks are global variables initialized in 

; consecFrom returns a list of consecutive values from idx up to N-1
(defun consecFrom(idx N)
	(if (< idx N)
		(cons idx (consecFrom (1+ idx) N))
		nil
	)
)

; setNth(list idx val) returns a new version of list with idx-th element set on val
(defun setNth(list idx val)
	(if (> idx 0)
		(cons (car list) (setNth (cdr list) (1- idx) val))
		(cons val (cdr list))
	)
)

; Checks if the provided index might be a valid element index
(defun validateIndex(idx)
	(if (and (integerp idx) (>= idx 0) (< idx N))
		t
		((lambda()
			(format t "Invalid index: ~d" idx)(terpri)
			nil
		))
	)
)

; parentOf(id prevAncestors) returns (parentId updatedAncestors) and allows looking for a parent without changing ancestors
(defun parentOf(id prevAncestors)
	(setq parentId (nth id prevAncestors))
	(if (= id parentId)
		(list id prevAncestors)
		((lambda()
			(setq parentId_ (nth parentId prevAncestors))
			(setq ancestors_ (setNth prevAncestors id parentId_))
			(parentOf parentId_ ancestors_)
		))
	)
)

; join(id1 id2) joins the 2 id-s and updates ancestors and ranks
(defun join(id1 id2)
	(format t "~d - ~d : " id1 id2)
	(when (and (validateIndex id1) (validateIndex id2))
		((lambda()
			(setq parent_ancestors (parentOf id1 ancestors))
			(setq parentId1 (car parent_ancestors))
			(setq ancestors_ (cadr parent_ancestors))
			(setq parent_ancestors (parentOf id2 ancestors_))
			(setq parentId2 (car parent_ancestors))
			(setq ancestors_ (cadr parent_ancestors))
			(if (= parentId1 parentId2)
				(setq ancestors ancestors_) ; id1 and id2 were already members of same parent; the ancestors can still be updated
				((lambda()
					(setq rank1 (nth parentId1 ranks))
					(setq rank2 (nth parentId2 ranks))
					(if (< rank1 rank2)
						(setq ancestors (setNth ancestors_ parentId1 parentId2)) ; set ancestor of parentId1 to parentId2
						(setq ancestors (setNth ancestors_ parentId2 parentId1)) ; set ancestor of parentId2 to parentId1
					)
					(if (= rank1 rank2)
						(setq ranks (setNth ranks parentId1 (1+ rank1))) ; increment rank for parentId1
					)
				))
			)
			(showUF)
		))
	)
)

; showUF() displays current groups based on ancestors
(defun showUF()
	(setq parents nil)
	(setq members nil)
	(dolist (id ids)
		(setq parent (car (parentOf id ancestors)))
		(setq parentPos (position parent parents))
		(if (null parentPos)
			((lambda()
				(setq parents (append parents (list parent)))
				(setq members (append members (list (list id))))
			))
			(setq members (setNth members parentPos (append (nth parentPos members) (list id))))
		)
	)
	(setq mapping (mapcar 'list parents members))
	(format t "~d groups: ~a" (length mapping) mapping)(terpri)
	(when (= (length mapping) 1) (write-line "All elements are now connected!"))
)

; Reads all the tokens from a string separated by spaces
(defun getTokens(line)
	(with-input-from-string (inp line)
		(loop for x = (read inp nil nil) while x collect x))
)

; Parsing the scenario file and executing it step by step
(with-open-file (stream "testScenario.txt" :direction :input :if-does-not-exist :error)
	(block ParsingBlock
		(do (
				(line (read-line stream nil) (read-line stream nil))
				(itemsCount nil)
				(validScenario t)
			)
			((null line))
			(when (and (string/= line "") (char/= (char line 0) #\#)) ; Ignoring empty lines or lines containing comments (starting with '#')
				(if (null itemsCount)
					((lambda()	; initialize itemsCount and the Union Find object's fields
						(setq itemsCount (getTokens line))
						(when (or (/= (length itemsCount) 1) (not (integerp (car itemsCount))))
							(return-from ParsingBlock (write-line "Couldn't read the items count!")))
						(setq itemsCount (car itemsCount))

						; global variables
						(defvar N itemsCount)
						(defvar ids (consecFrom 0 N))
						(defvar ancestors ids)
						(defvar ranks (make-list N :initial-element 0))

						(showUF)

						(when (< itemsCount 2) (write-line "Note that this problem makes sense only for at least 2 elements!"))
					))
					((lambda()	; read a new pair of indices to join
						(setq elemPair (getTokens line))
						(when (/= (length elemPair) 2)
							(return-from ParsingBlock (write-line "Read line containing less/more than 2 indices!")))
	  					(setq idx1 (car elemPair))
						(setq idx2 (cadr elemPair))
						(when (or (not (integerp idx1)) (not (integerp idx2)))
							(return-from ParsingBlock (write-line "Read a pair of indices with non integer value(s)!")))
						(join idx1 idx2)
					))
				)
			)
		)
	)
)

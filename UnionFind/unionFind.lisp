;
; Implementation of the UnionFind data structure described here:
; 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
;
; @2017 Florin Tulba (florintulba@yahoo.com)
;


; Count of items to be grouped
(defconstant N 10)

; consecFrom returns a list of consecutive values from idx up to N-1
(defun consecFrom(idx)
	(if (< idx N)
		(cons idx (consecFrom (1+ idx)))
		nil
	)
)

; global variables
(defvar ids (consecFrom 0))
(defvar ancestors ids)
(defvar ranks (make-list N :initial-element 0))

; setNth(list idx val) returns a new version of list with idx-th element set on val
(defun setNth(list idx val)
	(if (> idx 0)
		(cons (car list) (setNth (cdr list) (1- idx) val))
		(cons val (cdr list))
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
	t
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
)

(write-line "Initial uf:")(showUF)

(join 0 3)(showUF)
(join 4 5)(showUF)
(join 1 9)(showUF)
(join 2 8)(showUF)
(join 7 4)(showUF)
(join 9 0)(showUF)
(join 7 8)(showUF)
(join 1 6)(showUF)
(join 0 5)(showUF)

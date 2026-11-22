package pt

import (
	"fmt"
	"math"
	"sort"
	"sync/atomic"
	"time"
)

// SplitStrategy selects the heuristic used to place k-d tree split planes.
type SplitStrategy int

const (
	// SplitSAH builds the tree with a binned surface area heuristic. The
	// cost of a split is estimated from the surface areas of the two child
	// bounds and the number of shape references landing on each side, with
	// overlapping bounding boxes counted on both sides. This keeps
	// reference duplication (and therefore intersection candidates) low
	// when many shapes overlap.
	SplitSAH SplitStrategy = iota
	// SplitMedian reproduces the historical median-of-bounds builder. It
	// is retained as a fallback and for comparisons.
	SplitMedian
)

// TreeConfig controls tree construction. The zero value is valid: SAH is
// used with the documented defaults.
type TreeConfig struct {
	// Strategy selects the split heuristic. Defaults to SplitSAH.
	Strategy SplitStrategy
	// LeafCapacity is the maximum number of shapes stored in a leaf; nodes
	// at or below this size are not split. Defaults to 8.
	LeafCapacity int
	// MaxDepth caps the tree depth. Zero selects 8*ceil(log2(n+1)), capped
	// at 64. Explicit values are honored as given.
	MaxDepth int
	// SAHBins is the number of spatial bins used per axis by the binned
	// SAH. Defaults to 16.
	SAHBins int
	// SAHTraversal is the estimated cost of traversing an interior node,
	// relative to one shape intersection test (fixed at 1). The classic
	// ratio for k-d trees is 1:8; lower values favor splitting more
	// aggressively. Defaults to 0.125.
	SAHTraversal float64
	// SAHEmptyBonus discounts the reference cost for an empty child. It is
	// clamped to [0, 1]; zero disables the bonus. Defaults to 0.5.
	SAHEmptyBonus float64
	// CountCandidates, when true, makes every read-only Intersect count the
	// leaf shape tests into TreeStats via an atomic counter. It stays nil
	// (zero overhead) when false.
	CountCandidates bool
}

// DefaultTreeConfig returns the configuration used by NewTree.
func DefaultTreeConfig() TreeConfig {
	return TreeConfig{
		Strategy:        SplitSAH,
		LeafCapacity:    8,
		MaxDepth:        0,
		SAHBins:         16,
		SAHTraversal:    0.125,
		SAHEmptyBonus:   0.5,
		CountCandidates: false,
	}
}

func (c TreeConfig) withDefaults(n int) TreeConfig {
	d := DefaultTreeConfig()
	if c.LeafCapacity <= 0 {
		c.LeafCapacity = d.LeafCapacity
	}
	if c.SAHBins <= 0 {
		c.SAHBins = d.SAHBins
	}
	if c.SAHTraversal <= 0 || math.IsNaN(c.SAHTraversal) || math.IsInf(c.SAHTraversal, 0) {
		c.SAHTraversal = d.SAHTraversal
	}
	if math.IsNaN(c.SAHEmptyBonus) || c.SAHEmptyBonus < 0 {
		c.SAHEmptyBonus = d.SAHEmptyBonus
	}
	if c.SAHEmptyBonus > 1 {
		c.SAHEmptyBonus = 1
	}
	if c.MaxDepth <= 0 {
		maxDepth := int(math.Ceil(8 * math.Log2(float64(n)+1)))
		if maxDepth > 64 {
			maxDepth = 64
		}
		if maxDepth < 1 {
			maxDepth = 1
		}
		c.MaxDepth = maxDepth
	}
	return c
}

// TreeStats summarizes a built tree and, when candidate counting is enabled,
// the work performed by read-only intersection queries.
type TreeStats struct {
	Nodes          int           // total nodes (interior + leaves)
	InteriorNodes  int           // nodes with a split plane
	LeafNodes      int           // nodes holding shapes
	MaxDepth       int           // deepest leaf depth, root leaf is depth 0
	ShapeRefs      int           // shapes stored across all leaves (>= input count with overlap)
	DuplicateRefs  int           // ShapeRefs minus the input shape count
	LeafShapes     []int         // distribution of leaf sizes
	RootBoxArea    float64       // surface area of the scene bounding box
	LeafBoxArea    float64       // summed surface area of leaf bounding boxes
	TotalBoxArea   float64       // summed surface area of every node's bounds
	SAHEvaluations int           // split decisions that ran the SAH cost model
	SAHSplits      int           // split planes accepted from the SAH
	MedianSplits   int           // split planes placed by the median fallback
	BuildDuration  time.Duration // wall-clock time of the build
}

// Candidates returns the number of shape intersection tests performed since
// the counter was last reset. It is zero unless CountCandidates was enabled.
func (tree *Tree) Candidates() uint64 {
	if tree.candidates == nil {
		return 0
	}
	return atomic.LoadUint64(tree.candidates)
}

// ResetCandidates zeroes the runtime shape-test counter.
func (tree *Tree) ResetCandidates() {
	if tree.candidates != nil {
		atomic.StoreUint64(tree.candidates, 0)
	}
}

// Stats returns build-time statistics for the tree.
func (tree *Tree) Stats() TreeStats {
	return tree.stats
}

type Tree struct {
	Box        Box
	Root       *Node
	config     TreeConfig
	stats      TreeStats
	candidates *uint64
}

// NewTree builds a tree with the default SAH configuration.
func NewTree(shapes []Shape) *Tree {
	return NewTreeWithConfig(shapes, DefaultTreeConfig())
}

// NewTreeWithConfig builds a tree under the given configuration. The empty
// scene and single-shape cases produce trivial leaves without invoking a
// split heuristic; non-finite bounds (infinite primitives, NaNs) and
// degenerate axes fall back to median splits and, if necessary, leaves.
func NewTreeWithConfig(shapes []Shape, config TreeConfig) *Tree {
	fmt.Printf("Building k-d tree (%d shapes)... ", len(shapes))
	start := time.Now()
	box := BoxForShapes(shapes)
	config = config.withDefaults(len(shapes))
	tree := &Tree{Box: box, config: config}
	if config.CountCandidates {
		var c uint64
		tree.candidates = &c
	}
	if len(shapes) == 0 {
		// Bypass the heuristic entirely but still count the root leaf so
		// Nodes == InteriorNodes + LeafNodes holds for every input.
		tree.stats.Nodes++
		tree.stats.MaxDepth = 0
		tree.Root = tree.makeLeaf(shapes, 0)
	} else {
		tree.Root = tree.build(shapes, box, 0)
	}
	tree.stats.DuplicateRefs = tree.stats.ShapeRefs - len(shapes)
	tree.stats.BuildDuration = time.Since(start)
	tree.stats.RootBoxArea = box.SurfaceArea()
	fmt.Println("OK")
	return tree
}

func (tree *Tree) build(shapes []Shape, bounds Box, depth int) *Node {
	tree.stats.Nodes++
	if depth > tree.stats.MaxDepth {
		tree.stats.MaxDepth = depth
	}
	n := len(shapes)

	// Leaf conditions: capacity reached, depth exhausted, or non-finite
	// bounds (e.g. an infinite Plane) that make plane placement undefined.
	if n <= tree.config.LeafCapacity || depth >= tree.config.MaxDepth || !bounds.IsFinite() {
		return tree.makeLeaf(shapes, depth)
	}

	var axis Axis
	var point float64
	var usedSAH bool
	switch tree.config.Strategy {
	case SplitMedian:
		axis, point = tree.medianSplit(shapes)
	default:
		tree.stats.SAHEvaluations++
		if a, p, ok := tree.sahSplit(shapes, bounds); ok {
			axis, point, usedSAH = a, p, true
		} else {
			// Degenerate axis or no cost-improving plane: fall back to the
			// median builder before giving up on splitting.
			axis, point = tree.medianSplit(shapes)
		}
	}
	if axis == AxisNone {
		return tree.makeLeaf(shapes, depth)
	}

	left, right := tree.partition(shapes, axis, point)
	// A split that leaves every reference on one side cannot make progress;
	// store the node as a leaf to guarantee termination on coincident boxes.
	if len(left) == 0 || len(right) == 0 || len(left) >= n || len(right) >= n {
		return tree.makeLeaf(shapes, depth)
	}

	if usedSAH {
		tree.stats.SAHSplits++
	} else {
		tree.stats.MedianSplits++
	}
	tree.stats.InteriorNodes++

	leftBox := BoxForShapes(left)
	rightBox := BoxForShapes(right)
	node := &Node{Axis: axis, Point: point}
	node.Left = tree.build(left, leftBox, depth+1)
	node.Right = tree.build(right, rightBox, depth+1)
	tree.stats.TotalBoxArea += bounds.SurfaceArea()
	return node
}

func (tree *Tree) makeLeaf(shapes []Shape, depth int) *Node {
	tree.stats.LeafNodes++
	tree.stats.LeafShapes = append(tree.stats.LeafShapes, len(shapes))
	tree.stats.ShapeRefs += len(shapes)
	area := BoxForShapes(shapes).SurfaceArea()
	tree.stats.LeafBoxArea += area
	tree.stats.TotalBoxArea += area
	node := &Node{Axis: AxisNone, Shapes: shapes}
	if tree.candidates != nil {
		node.candidates = tree.candidates
	}
	return node
}

// sahBin accumulates the bounds of the shapes whose centroids fall into a
// spatial bin along one axis.
type sahBin struct {
	count int
	box   Box
}

// sahSplit evaluates SplitSAHBins planes on each axis using the binned
// surface area heuristic and returns the cheapest plane that beats making
// the node a leaf. Overlapping boxes are counted on both sides, matching the
// actual partition semantics.
func (tree *Tree) sahSplit(shapes []Shape, bounds Box) (Axis, float64, bool) {
	binsN := tree.config.SAHBins
	traversal := tree.config.SAHTraversal
	emptyBonus := tree.config.SAHEmptyBonus
	// A leaf performs one intersection test per reference; a split pays one
	// traversal plus the area-weighted reference work of both children.
	leafCost := float64(len(shapes))
	bestCost := leafCost
	bestAxis := AxisNone
	bestPoint := 0.0
	size := bounds.Size()
	// SAH probabilities are child areas normalized by the parent area;
	// without this division the raw area magnitude dominates the cost and
	// every split looks worse than a leaf. A zero-area parent has no
	// well-defined probabilities, so the caller falls back to the median.
	parentArea := bounds.SurfaceArea()
	if parentArea <= 0 {
		return AxisNone, 0, false
	}

	// Precompute each shape's bounds once; they are reused on every axis.
	type shapeBounds struct {
		box    Box
		min    [3]float64
		max    [3]float64
		center [3]float64
	}
	entries := make([]shapeBounds, len(shapes))
	for i, shape := range shapes {
		box := shape.BoundingBox()
		c := box.Center()
		entries[i] = shapeBounds{
			box:    box,
			min:    [3]float64{box.Min.X, box.Min.Y, box.Min.Z},
			max:    [3]float64{box.Max.X, box.Max.Y, box.Max.Z},
			center: [3]float64{c.X, c.Y, c.Z},
		}
	}

	axisIndex := func(a Axis) int {
		switch a {
		case AxisX:
			return 0
		case AxisY:
			return 1
		default:
			return 2
		}
	}

	for _, axis := range []Axis{AxisX, AxisY, AxisZ} {
		ai := axisIndex(axis)
		extent := axisComponent(size, axis)
		// Degenerate axis: plane placement is undefined here, skip it.
		if extent <= 0 || math.IsNaN(extent) || math.IsInf(extent, 0) {
			continue
		}
		bins := make([]sahBin, binsN)
		lo := axisComponent(bounds.Min, axis)
		invExtent := 1 / extent
		for i := range entries {
			idx := int(float64(binsN) * (entries[i].center[ai] - lo) * invExtent)
			if idx < 0 {
				idx = 0
			} else if idx >= binsN {
				idx = binsN - 1
			}
			b := &bins[idx]
			if b.count == 0 {
				b.box = entries[i].box
			} else {
				b.box = b.box.Extend(entries[i].box)
			}
			b.count++
		}

		// Prefix/suffix boxes estimate child areas for every candidate plane
		// between bins k and k+1.
		leftBoxes := make([]Box, binsN-1)
		rightBoxes := make([]Box, binsN-1)
		{
			acc := Box{}
			init := false
			for k := 0; k < binsN-1; k++ {
				if bins[k].count > 0 {
					if !init {
						acc, init = bins[k].box, true
					} else {
						acc = acc.Extend(bins[k].box)
					}
				}
				leftBoxes[k] = acc
			}
			acc, init = Box{}, false
			for k := binsN - 1; k >= 1; k-- {
				if bins[k].count > 0 {
					if !init {
						acc, init = bins[k].box, true
					} else {
						acc = acc.Extend(bins[k].box)
					}
				}
				rightBoxes[k-1] = acc
			}
		}

		for k := 0; k < binsN-1; k++ {
			point := lo + extent*float64(k+1)/float64(binsN)
			// Exact partition counts: a straddling box is referenced by both
			// children, so it is counted on each side. Bins only estimate
			// areas; using centroid counts here would ignore duplication and
			// make overlapping scenes reject every useful split.
			lc, rc := 0, 0
			for i := range entries {
				if entries[i].min[ai] <= point {
					lc++
				}
				if entries[i].max[ai] >= point {
					rc++
				}
			}
			if lc == 0 && rc == 0 {
				continue
			}
			la, ra := 0.0, 0.0
			if lc > 0 {
				la = leftBoxes[k].SurfaceArea() / parentArea
			}
			if rc > 0 {
				ra = rightBoxes[k].SurfaceArea() / parentArea
			}
			// Reference cost, discounted for empty children.
			lw, rw := 1.0, 1.0
			if lc == 0 {
				rw = 1 - emptyBonus
			}
			if rc == 0 {
				lw = 1 - emptyBonus
			}
			cost := traversal + float64(lc)*la*lw + float64(rc)*ra*rw
			if cost < bestCost {
				bestCost = cost
				bestAxis = axis
				bestPoint = point
			}
		}
	}

	if bestAxis == AxisNone {
		return AxisNone, 0, false
	}
	return bestAxis, bestPoint, true
}

// medianSplit reproduces the historical builder: the median of all shape
// bounds on each axis, keeping the axis whose larger partition is smallest.
// Returns AxisNone when no axis reduces the partition below the input size.
func (tree *Tree) medianSplit(shapes []Shape) (Axis, float64) {
	xs := make([]float64, 0, len(shapes)*2)
	ys := make([]float64, 0, len(shapes)*2)
	zs := make([]float64, 0, len(shapes)*2)
	for _, shape := range shapes {
		box := shape.BoundingBox()
		xs = append(xs, box.Min.X, box.Max.X)
		ys = append(ys, box.Min.Y, box.Max.Y)
		zs = append(zs, box.Min.Z, box.Max.Z)
	}
	sort.Float64s(xs)
	sort.Float64s(ys)
	sort.Float64s(zs)
	candidates := []struct {
		axis  Axis
		point float64
	}{
		{AxisX, Median(xs)},
		{AxisY, Median(ys)},
		{AxisZ, Median(zs)},
	}
	best := int(float64(len(shapes)) * 0.85)
	bestAxis := AxisNone
	bestPoint := 0.0
	for _, c := range candidates {
		score := tree.partitionScore(shapes, c.axis, c.point)
		if score < best {
			best = score
			bestAxis = c.axis
			bestPoint = c.point
		}
	}
	return bestAxis, bestPoint
}

func (tree *Tree) partitionScore(shapes []Shape, axis Axis, point float64) int {
	left, right := 0, 0
	for _, shape := range shapes {
		box := shape.BoundingBox()
		l, r := box.Partition(axis, point)
		if l {
			left++
		}
		if r {
			right++
		}
	}
	if left >= right {
		return left
	}
	return right
}

// partition assigns shapes to children. A shape whose bounding box straddles
// the plane is referenced by both children, which is why overlapping scenes
// inflate ShapeRefs and why the SAH accounts for both sides.
func (tree *Tree) partition(shapes []Shape, axis Axis, point float64) (left, right []Shape) {
	left = make([]Shape, 0, len(shapes))
	right = make([]Shape, 0, len(shapes))
	for _, shape := range shapes {
		box := shape.BoundingBox()
		l, r := box.Partition(axis, point)
		if l {
			left = append(left, shape)
		}
		if r {
			right = append(right, shape)
		}
	}
	return
}

func axisComponent(v Vector, axis Axis) float64 {
	switch axis {
	case AxisX:
		return v.X
	case AxisY:
		return v.Y
	default:
		return v.Z
	}
}

func (tree *Tree) Intersect(r Ray) Hit {
	tmin, tmax := tree.Box.Intersect(r)
	if tmax < tmin || tmax <= 0 {
		return NoHit
	}
	return tree.Root.Intersect(r, tmin, tmax)
}

type Node struct {
	Axis       Axis
	Point      float64
	Shapes     []Shape
	Left       *Node
	Right      *Node
	candidates *uint64
}

func (node *Node) Intersect(r Ray, tmin, tmax float64) Hit {
	var tsplit float64
	var leftFirst bool
	switch node.Axis {
	case AxisNone:
		return node.IntersectShapes(r)
	case AxisX:
		tsplit = (node.Point - r.Origin.X) / r.Direction.X
		leftFirst = (r.Origin.X < node.Point) || (r.Origin.X == node.Point && r.Direction.X <= 0)
	case AxisY:
		tsplit = (node.Point - r.Origin.Y) / r.Direction.Y
		leftFirst = (r.Origin.Y < node.Point) || (r.Origin.Y == node.Point && r.Direction.Y <= 0)
	case AxisZ:
		tsplit = (node.Point - r.Origin.Z) / r.Direction.Z
		leftFirst = (r.Origin.Z < node.Point) || (r.Origin.Z == node.Point && r.Direction.Z <= 0)
	}
	var first, second *Node
	if leftFirst {
		first = node.Left
		second = node.Right
	} else {
		first = node.Right
		second = node.Left
	}
	if tsplit > tmax || tsplit <= 0 {
		return first.Intersect(r, tmin, tmax)
	} else if tsplit < tmin {
		return second.Intersect(r, tmin, tmax)
	} else {
		h1 := first.Intersect(r, tmin, tsplit)
		if h1.T <= tsplit {
			return h1
		}
		h2 := second.Intersect(r, tsplit, math.Min(tmax, h1.T))
		if h1.T <= h2.T {
			return h1
		}
		return h2
	}
}

func (node *Node) IntersectShapes(r Ray) Hit {
	if node.candidates != nil {
		atomic.AddUint64(node.candidates, uint64(len(node.Shapes)))
	}
	hit := NoHit
	for _, shape := range node.Shapes {
		h := shape.Intersect(r)
		if h.T < hit.T {
			hit = h
		}
	}
	return hit
}

// String renders build statistics in a human-readable form.
func (s TreeStats) String() string {
	avg := 0.0
	if s.LeafNodes > 0 {
		avg = float64(s.ShapeRefs) / float64(s.LeafNodes)
	}
	return fmt.Sprintf(
		"nodes=%d (interior=%d leaves=%d maxDepth=%d) refs=%d duplicates=%d "+
			"avgLeafShapes=%.2f rootArea=%.4g leafArea=%.4g totalBoxArea=%.4g "+
			"sah(evals=%d accepted=%d) medianFallback=%d build=%s",
		s.Nodes, s.InteriorNodes, s.LeafNodes, s.MaxDepth,
		s.ShapeRefs, s.DuplicateRefs, avg,
		s.RootBoxArea, s.LeafBoxArea, s.TotalBoxArea,
		s.SAHEvaluations, s.SAHSplits, s.MedianSplits,
		s.BuildDuration.String())
}

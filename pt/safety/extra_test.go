package safety

import (
	"errors"
	"testing"
	"time"

	"github.com/fogleman/pt/pt"
)

// TestRejectionErrorUnwrap proves rejection errors satisfy errors.Is against
// the sentinel and carry the failing stage.
func TestRejectionErrorUnwrap(t *testing.T) {
	rep, err := CheckOBJBytes([]byte("v 0 0 0\nf 1 2 3\n"), DefaultLimits())
	if err == nil {
		t.Fatal("expected rejection")
	}
	if !errors.Is(err, ErrRejected) {
		t.Fatalf("rejection does not unwrap to ErrRejected: %v", err)
	}
	var rej *RejectionError
	if !errors.As(err, &rej) {
		t.Fatal("error is not a *RejectionError")
	}
	if rej.Stage != StageIndex || rej.Error() == "" {
		t.Fatalf("malformed rejection: %+v", rej)
	}
	if rep.Err != err {
		t.Fatal("report did not retain the rejection error")
	}
}

// TestMissingFileStageStat confirms a non-existent file is reported at
// StageStat rather than panicking.
func TestMissingFileStageStat(t *testing.T) {
	rep, err := CheckOBJ("definitely-missing-12345.obj", DefaultLimits())
	if err == nil {
		t.Fatal("missing file was accepted")
	}
	if rep.Stage != StageStat {
		t.Fatalf("expected StageStat, got %s", rep.Stage)
	}
	rep, err = CheckSTL("definitely-missing-12345.stl", DefaultLimits())
	if err == nil || rep.Stage != StageStat {
		t.Fatalf("STL missing file: expected StageStat, got %s", rep.Stage)
	}
}

// TestZeroLimitsFallBackDefaults exercises the Limits.fill() fallback path.
func TestZeroLimitsFallBackDefaults(t *testing.T) {
	rep, err := CheckOBJBytes([]byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), Limits{})
	if err != nil || !rep.OK() {
		t.Fatalf("zero-valued limits did not fall back to defaults: %s %v", rep.Stage, err)
	}
	g := NewGuardedSDF(unitSphere{}, 0)
	if g.Limit() != DefaultLimits().MaxSDFDepth {
		t.Fatalf("expected default depth %d, got %d", DefaultLimits().MaxSDFDepth, g.Limit())
	}
}

// TestBoundedSDFShapeAccessors covers the Shape surface methods and the
// NormalAt finite/non-finite branches.
func TestBoundedSDFShapeAccessors(t *testing.T) {
	shape := NewBoundedSDFShape(GuardTree(unitSphere{}, 64), pt.DiffuseMaterial(pt.HexColor(0x112233)), DefaultLimits())
	if shape.BoundingBox() == (pt.Box{}) {
		t.Fatal("bounding box not forwarded")
	}
	shape.Compile() // must be a safe no-op
	if got := shape.UV(pt.Vector{}); got != (pt.Vector{}) {
		t.Fatalf("UV should be zero vector, got %v", got)
	}
	mat := shape.MaterialAt(pt.Vector{})
	want := pt.DiffuseMaterial(pt.HexColor(0x112233))
	if mat != want {
		t.Fatalf("MaterialAt mismatch: %+v vs %+v", mat, want)
	}
	// NormalAt on a benign sphere at an on-surface point must be finite.
	n := shape.NormalAt(pt.Vector{X: 0.5, Y: 0, Z: 0})
	if !finite(n.X) || !finite(n.Y) || !finite(n.Z) {
		t.Fatalf("normal not finite: %v", n)
	}
}

// TestNormalAtNonFiniteFallback forces a NaN-producing SDF through NormalAt
// and asserts the deterministic fallback normal instead of a NaN vector.
func TestNormalAtNonFiniteFallback(t *testing.T) {
	shape := NewBoundedSDFShape(GuardTree(nanSDF{}, 64), pt.DiffuseMaterial(pt.White), DefaultLimits())
	n := shape.NormalAt(pt.Vector{})
	if n != (pt.Vector{X: 0, Y: 0, Z: 1}) {
		t.Fatalf("expected +Z fallback normal, got %v", n)
	}
}

// TestSDFDepthAllCompositeTypes walks every pt composite wrapper so the
// structural checker and GuardTree cover the full type switch.
func TestSDFDepthAllCompositeTypes(t *testing.T) {
	sphere := pt.NewSphereSDF(0.5)
	cube := pt.NewCubeSDF(pt.Vector{X: 1, Y: 1, Z: 1})
	cyl := pt.NewCylinderSDF(0.5, 1)
	deep := pt.NewTransformSDF(
		pt.NewScaleSDF(
			pt.NewDifferenceSDF(
				pt.NewIntersectionSDF(sphere, cube),
				cyl),
			2),
		pt.Identity())
	limits := DefaultLimits()
	rep, err := CheckSDFDepth(deep, limits)
	if err != nil {
		t.Fatalf("composite tree rejected unexpectedly at %s: %v", rep.Stage, err)
	}
	if rep.Vertices < 4 {
		t.Fatalf("depth measurement too shallow: %d", rep.Vertices)
	}
	// GuardTree must accept and wrap the mixed composite tree.
	guarded := GuardTree(deep, 32)
	if d := guarded.Evaluate(pt.Vector{X: 0.1, Y: 0.1, Z: 0.1}); d != d {
		t.Fatal("wrapped composite returned NaN")
	}
}

// TestGuardedSDFInfBudgetExhaustion directly forces the depth budget to zero
// and confirms evaluation returns +Inf rather than descending.
func TestGuardedSDFInfBudgetExhaustion(t *testing.T) {
	g := NewGuardedSDF(unitSphere{}, 1)
	g.budget.remaining = 0
	if d := g.Evaluate(pt.Vector{}); !isInf(d) {
		t.Fatalf("expected +Inf on exhausted budget, got %g", d)
	}
}

// TestBoundedSDFBadBoxIntersection feeds rays whose box intersection is
// non-finite to confirm the early NoHit guard.
func TestBoundedSDFBadBoxIntersection(t *testing.T) {
	// A ray with a zero direction yields NaN box t-values.
	shape := NewBoundedSDFShape(GuardTree(unitSphere{}, 64), pt.DiffuseMaterial(pt.White), DefaultLimits())
	ray := pt.Ray{Origin: pt.Vector{}, Direction: pt.Vector{}}
	if h := shape.Intersect(ray); h.Ok() {
		t.Fatal("zero-direction ray produced a hit")
	}
}

// TestTruncatedBinarySTL forges a header whose count is internally consistent
// with the declared size but whose record bytes are cut short, exercising the
// io.ReadFull truncation branch.
func TestTruncatedBinarySTL(t *testing.T) {
	data := buildValidBinarySTL() // 2 complete records, 84+100 bytes
	truncated := data[:84+50+20]  // second record cut off
	rep, err := CheckSTLBytes(truncated, DefaultLimits())
	if err == nil || rep.Stage != StageHeader {
		t.Fatalf("expected StageHeader for truncation, got %s err=%v", rep.Stage, err)
	}
}

// TestOversizeFileQuota covers the file-based (not byte-based) size gate.
func TestOversizeFileQuota(t *testing.T) {
	limits := DefaultLimits()
	limits.MaxFileSize = 16
	path := writeSample(t, []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), ".obj")
	rep, err := CheckOBJ(path, limits)
	if err == nil || rep.Stage != StageSize {
		t.Fatalf("expected StageSize, got %s err=%v", rep.Stage, err)
	}
}

// TestCheckMeshNilMesh covers the nil-mesh defensive branch.
func TestCheckMeshNilMesh(t *testing.T) {
	rep := &Report{Format: "obj"}
	if _, err := CheckMesh(nil, rep, DefaultLimits()); err == nil || rep.Stage != StagePanic {
		t.Fatalf("expected StagePanic for nil mesh, got %s", rep.Stage)
	}
}

// TestMarchFromInsideSurface starts a ray inside the sphere's bounding box so
// the first signed distance is negative, exercising the jump/backstep branch
// of the bounded marcher while still converging to a finite exit.
func TestMarchFromInsideSurface(t *testing.T) {
	shape := NewBoundedSDFShape(GuardTree(unitSphere{}, 64), pt.DiffuseMaterial(pt.White), DefaultLimits())
	// Origin inside the radius-0.5 sphere, marching outward.
	ray := pt.Ray{
		Origin:    pt.Vector{X: 0, Y: 0, Z: 0},
		Direction: pt.Vector{X: 1, Y: 0, Z: 0},
	}
	h := shape.Intersect(ray)
	// Starting inside, the marcher should still terminate bounded; whether it
	// reports a hit is implementation-defined, but it must not hang or NaN.
	if h.T != h.T {
		t.Fatal("inside-surface march returned NaN distance")
	}
}

// bigStallSDF is a non-converging SDF inside an enormous bounding box, so the
// marcher cannot leave the box quickly and must rely on its deadline.
type bigStallSDF struct{}

func (bigStallSDF) Evaluate(pt.Vector) float64 { return sdfEpsilon * 4 }
func (bigStallSDF) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -1e8, Y: -1e8, Z: -1e8}, Max: pt.Vector{X: 1e8, Y: 1e8, Z: 1e8}}
}

// TestSDFShapeDeadlineFires lowers the deadline and uses a large-box stall SDF
// to prove the wall-clock branch is reachable, not just the iteration cap.
func TestSDFShapeDeadlineFires(t *testing.T) {
	limits := DefaultLimits()
	limits.SampleTimeout = 20 * time.Millisecond
	shape := NewBoundedSDFShape(GuardTree(bigStallSDF{}, 64), pt.DiffuseMaterial(pt.White), limits)
	shape.MaxIterations = 1_000_000_000 // force the deadline, not the iteration cap
	ray := pt.Ray{
		Origin:    pt.Vector{X: 0, Y: 0, Z: 0},
		Direction: pt.Vector{X: 1, Y: 0, Z: 0},
	}
	h := shape.Intersect(ray)
	if h.Ok() {
		t.Fatal("stalling SDF produced a false hit under deadline")
	}
	if shape.Timeouts == 0 {
		t.Fatal("deadline termination was not recorded")
	}
}

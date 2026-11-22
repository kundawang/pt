package safety

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fogleman/pt/pt"
)

func newRand(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

func mathFloat32NaN() uint32 { return math.Float32bits(float32(math.NaN())) }

// buildValidBinarySTL emits a spec-correct 2-triangle binary STL with known
// finite coordinates (independent of the randomized, possibly-hostile
// generator used elsewhere).
func buildValidBinarySTL() []byte {
	var b bytes.Buffer
	var header [80]byte
	b.Write(header[:])
	_ = binary.Write(&b, binary.LittleEndian, uint32(2))
	tris := [2][3][3]float32{
		{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
		{{0, 0, 1}, {1, 0, 1}, {1, 1, 1}},
	}
	for _, tri := range tris {
		rec := make([]byte, 50)
		for v := 0; v < 3; v++ {
			for axis := 0; axis < 3; axis++ {
				binary.LittleEndian.PutUint32(rec[12+v*12+axis*4:],
					math.Float32bits(tri[v][axis]))
			}
		}
		b.Write(rec)
	}
	return b.Bytes()
}

// writeSample copies an in-memory sample into its own empty temp directory,
// honoring the untrusted-data handling rule: generated samples never share a
// directory with test sources or each other.
func writeSample(t *testing.T, data []byte, suffix string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "sample"+suffix)
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestSafeLoadOBJRejectsOOB is the core fault-injection proof. The upstream
// pt.LoadOBJ demonstrably panics on this input (see probe in the design
// notes); SafeLoadOBJ must convert that into a StageIndex rejection from the
// independent checker, and the real loader must never run.
func TestSafeLoadOBJRejectsOOB(t *testing.T) {
	path := writeSample(t, []byte("v 0 0 0\nf 1 5 9\n"), ".obj")
	mesh, rep, err := SafeLoadOBJ(path, pt.DiffuseMaterial(pt.White), DefaultLimits())
	if err == nil {
		t.Fatal("malicious OBJ was loaded instead of rejected")
	}
	if mesh != nil {
		t.Fatal("non-nil mesh returned on rejection")
	}
	if rep.Stage != StageIndex {
		t.Fatalf("expected StageIndex, got %s: %v", rep.Stage, err)
	}
}

// TestSafeLoadOBJShortVertexPanicContainment injects the truncated "v 1"
// input that panics inside pt.LoadOBJ. Even with the checker bypassed, the
// panic guard must convert it into StagePanic rather than crashing the test.
func TestSafeLoadOBJShortVertexPanicContainment(t *testing.T) {
	// This sample is rejected by the checker at StageHeader first. To prove
	// the panic boundary independently, call the guarded loader directly with
	// a checker-passing-but-loader-hostile shape is not possible from outside
	// the package, so instead assert the public path still fails safely and
	// additionally exercise guarded() with a function that panics.
	path := writeSample(t, []byte("v 1\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), ".obj")
	_, rep, err := SafeLoadOBJ(path, pt.DiffuseMaterial(pt.White), DefaultLimits())
	if err == nil || rep.Stage != StageHeader {
		t.Fatalf("expected StageHeader, got %s err=%v", rep.Stage, err)
	}

	// Direct panic-injection into the guarded boundary.
	report := &Report{Format: "obj"}
	_, perr := guarded(func() (*pt.Mesh, error) {
		panic("simulated loader index out of range")
	}, DefaultLimits(), report)
	if perr == nil || report.Stage != StagePanic {
		t.Fatalf("panic not contained: stage=%s err=%v", report.Stage, perr)
	}
}

// TestSafeLoadOBJValidRoundTrip confirms a clean sample traverses all three
// pipeline stages and yields a usable, finite mesh with the expected counts.
func TestSafeLoadOBJValidRoundTrip(t *testing.T) {
	path := writeSample(t, mustReadFile(t, testdata(t, "regression-cube.obj")), ".obj")
	mesh, rep, err := SafeLoadOBJ(path, pt.DiffuseMaterial(pt.White), DefaultLimits())
	if err != nil {
		t.Fatalf("valid cube rejected at %s: %v", rep.Stage, err)
	}
	if len(mesh.Triangles) != 12 {
		t.Fatalf("expected 12 triangles, got %d", len(mesh.Triangles))
	}
	if !rep.OK() || rep.Stage != StageAccepted {
		t.Fatalf("unexpected report: %+v", rep)
	}
	mesh.Compile() // k-d tree must build without panic on validated geometry
}

// TestSafeLoadSTLCorruptHeader proves the binary count-amplification attack is
// rejected before loadSTLB can pre-allocate a slice sized by the forged count.
func TestSafeLoadSTLCorruptHeader(t *testing.T) {
	data := genSTLBinary(newRand(9), 2, 1_000_000_000, true)
	path := writeSample(t, data, ".stl")
	mesh, rep, err := SafeLoadSTL(path, pt.DiffuseMaterial(pt.White), DefaultLimits())
	if err == nil {
		t.Fatal("forged-count STL accepted")
	}
	if mesh != nil {
		t.Fatal("non-nil mesh on rejection")
	}
	if rep.Stage != StageHeader {
		t.Fatalf("expected StageHeader, got %s", rep.Stage)
	}
}

// TestSafeLoadSTLValidBinary builds a correct binary STL and confirms it
// survives both the reference checker and the real loader.
func TestSafeLoadSTLValidBinary(t *testing.T) {
	data := buildValidBinarySTL()
	path := writeSample(t, data, ".stl")
	mesh, rep, err := SafeLoadSTL(path, pt.DiffuseMaterial(pt.White), DefaultLimits())
	if err != nil {
		t.Fatalf("valid binary STL rejected at %s: %v", rep.Stage, err)
	}
	if len(mesh.Triangles) != 2 {
		t.Fatalf("expected 2 triangles, got %d", len(mesh.Triangles))
	}
}

// TestSafeLoaderTimeout injects a loader that blocks forever and asserts the
// deadline returns StageTimeout instead of hanging the suite.
func TestSafeLoaderTimeout(t *testing.T) {
	limits := DefaultLimits()
	limits.SampleTimeout = 100 * time.Millisecond
	report := &Report{Format: "obj", Stage: StageAccepted} // checker pre-passed
	start := time.Now()
	_, err := guarded(func() (*pt.Mesh, error) {
		time.Sleep(5 * time.Second) // simulate a wedged loader
		return nil, nil
	}, limits, report)
	elapsed := time.Since(start)
	if err == nil || report.Stage != StageTimeout {
		t.Fatalf("expected StageTimeout, got %s err=%v", report.Stage, err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("timeout did not fire promptly: %v", elapsed)
	}
}

// TestDegenerateMeshRejection constructs a loader result that passes text
// checks but collapses geometrically, proving the post-load CheckMesh stage
// is a real independent gate.
func TestDegenerateMeshRejection(t *testing.T) {
	zero := pt.Vector{}
	tris := []*pt.Triangle{
		{V1: zero, V2: zero, V3: zero},
		{V1: zero, V2: pt.Vector{X: 1}, V3: pt.Vector{X: 1}}, // collinear
	}
	mesh := pt.NewMesh(tris)
	rep := &Report{Format: "obj", Stage: StageAccepted}
	if _, err := CheckMesh(mesh, rep, DefaultLimits()); err == nil {
		t.Fatal("fully degenerate mesh passed geometric check")
	} else if rep.Stage != StageDegenerate {
		t.Fatalf("expected StageDegenerate, got %s", rep.Stage)
	}
}

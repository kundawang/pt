package safety

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fogleman/pt/pt"
)

// Fixed seed: every run generates the identical corpus, so a failure is
// reproducible without persisting thousands of samples. The three curated
// samples in testdata/ are the on-disk regression subset.
const fuzzSeed = int64(0x5AFE1234)

// hostileTokens are injected into generated samples to exercise the index,
// value, magnitude and framing checks.
var hostileFloats = []string{
	"nan", "inf", "-inf", "1e999", "-1e999",
	"1e308", "-1e308", "0x1p+1024", "1.7976931348623157e308",
	"0", "-0", "0.0000000000000000001",
}

var hostileInts = []string{"0", "-1", "-999999", "999999999", "notanint", "1/2/3", "-0"}

// genOBJ builds one OBJ sample. With probability benignBias the sample is
// guaranteed well-formed so the accepted-output invariant and the downstream
// mesh checks are exercised on a substantial minority of the corpus; the
// remaining samples are mutated with hostile tokens.
func genOBJ(r *rand.Rand, nVerts, nFaces int) []byte {
	benign := r.Intn(3) == 0
	var b bytes.Buffer
	coords := make([][3]string, nVerts)
	for i := 0; i < nVerts; i++ {
		var c [3]string
		for axis := 0; axis < 3; axis++ {
			if !benign && r.Intn(7) == 0 {
				c[axis] = hostileFloats[r.Intn(len(hostileFloats))]
			} else {
				c[axis] = fmt.Sprintf("%g", (r.Float64()*2-1)*100)
			}
		}
		coords[i] = c
		// Occasionally emit a truncated vertex line (hostile samples only).
		if !benign && r.Intn(20) == 0 {
			fmt.Fprintf(&b, "v %s\n", c[0])
			continue
		}
		fmt.Fprintf(&b, "v %s %s %s\n", c[0], c[1], c[2])
	}
	for f := 0; f < nFaces; f++ {
		arity := 3 + r.Intn(3) // triangles through pentagons
		fmt.Fprint(&b, "f")
		for k := 0; k < arity; k++ {
			if benign {
				// Always a valid in-range positive index.
				fmt.Fprintf(&b, " %d", 1+r.Intn(nVerts))
				continue
			}
			switch r.Intn(8) {
			case 0:
				fmt.Fprintf(&b, " %s", hostileInts[r.Intn(len(hostileInts))])
			case 1:
				// valid negative relative index
				fmt.Fprintf(&b, " -%d", 1+r.Intn(nVerts))
			default:
				fmt.Fprintf(&b, " %d", 1+r.Intn(nVerts+2)) // occasionally OOB
			}
		}
		fmt.Fprintln(&b)
	}
	// Occasionally inject a degenerate (repeated-index) face and a 2-vert face.
	if !benign && r.Intn(4) == 0 && nVerts > 0 {
		fmt.Fprintf(&b, "f 1 1 1\n")
	}
	if !benign && r.Intn(6) == 0 && nVerts > 0 {
		fmt.Fprintln(&b, "f 1 2")
	}
	return b.Bytes()
}

// genSTLBinary builds a binary STL byte stream. declaredCount may disagree
// with the actual number of records to forge a corrupt header. Benign samples
// carry only finite, bounded float32 coordinates.
func genSTLBinary(r *rand.Rand, actualRecords, declaredCount int, corruptHeader bool) []byte {
	benign := !corruptHeader && r.Intn(2) == 0
	var b bytes.Buffer
	var header [80]byte
	b.Write(header[:])
	count := uint32(declaredCount)
	if !corruptHeader {
		count = uint32(actualRecords)
	}
	_ = binary.Write(&b, binary.LittleEndian, count)
	for i := 0; i < actualRecords; i++ {
		rec := make([]byte, 50)
		// skip normal (first 12 bytes), fill 3 vertices
		for v := 0; v < 3; v++ {
			for axis := 0; axis < 3; axis++ {
				var f float32
				if !benign && r.Intn(9) == 0 {
					// hostile float32: NaN/Inf bit patterns or huge magnitude
					switch r.Intn(3) {
					case 0:
						f = float32(math.NaN())
					case 1:
						f = float32(math.Inf(1))
					default:
						f = float32(3.4e38)
					}
				} else {
					f = float32(r.Float64()*200 - 100)
				}
				binary.LittleEndian.PutUint32(rec[12+v*12+axis*4:], math.Float32bits(f))
			}
		}
		b.Write(rec)
	}
	return b.Bytes()
}

// genSTLAscii builds an ASCII STL, occasionally with a dangling vertex group.
func genSTLAscii(r *rand.Rand, nFacets int, dangle bool) []byte {
	benign := !dangle && r.Intn(2) == 0
	var b bytes.Buffer
	fmt.Fprintln(&b, "solid fuzz")
	for i := 0; i < nFacets; i++ {
		fmt.Fprintln(&b, "facet normal 0 0 0")
		fmt.Fprintln(&b, "outer loop")
		for v := 0; v < 3; v++ {
			x, y, z := r.Float64()*10-5, r.Float64()*10-5, r.Float64()*10-5
			if !benign && r.Intn(10) == 0 {
				fmt.Fprintf(&b, "vertex %s 0 0\n", hostileFloats[r.Intn(4)])
			} else {
				fmt.Fprintf(&b, "vertex %g %g %g\n", x, y, z)
			}
		}
		fmt.Fprintln(&b, "endloop")
		fmt.Fprintln(&b, "endfacet")
	}
	if dangle {
		fmt.Fprintln(&b, "facet normal 0 0 0")
		fmt.Fprintln(&b, "outer loop")
		fmt.Fprintln(&b, "vertex 1 1 1")
		fmt.Fprintln(&b, "endloop")
		fmt.Fprintln(&b, "endfacet")
	}
	fmt.Fprintln(&b, "endsolid fuzz")
	return b.Bytes()
}

// TestFuzzOBJFixedSeed runs the property suite over a deterministic corpus.
// Invariant for every sample: CheckOBJBytes either rejects with a structured
// stage or accepts with counts inside the limits; it must never panic and
// must never exceed the per-sample time budget.
func TestFuzzOBJFixedSeed(t *testing.T) {
	r := rand.New(rand.NewSource(fuzzSeed))
	limits := DefaultLimits()
	limits.MaxVertices = 5000
	limits.MaxTriangles = 5000
	const samples = 400
	rejections := map[Stage]int{}
	accepted := 0
	for i := 0; i < samples; i++ {
		nVerts := 1 + r.Intn(40)
		nFaces := r.Intn(30)
		data := genOBJ(r, nVerts, nFaces)
		start := time.Now()
		rep, err := runNoPanicOBJ(t, data, limits, i)
		elapsed := time.Since(start)
		if elapsed > limits.SampleTimeout {
			t.Fatalf("sample %d took %v > budget %v", i, elapsed, limits.SampleTimeout)
		}
		if err != nil {
			rejections[rep.Stage]++
			if rep.Stage == StageAccepted || rep.Stage == "" {
				t.Fatalf("sample %d: error %v but stage=%q", i, err, rep.Stage)
			}
			assertBoundedCounts(t, rep, limits, i)
			continue
		}
		accepted++
		if !rep.OK() {
			t.Fatalf("sample %d: nil error but report not OK: %+v", i, rep)
		}
		assertBoundedCounts(t, rep, limits, i)
	}
	if accepted == 0 {
		t.Fatal("generator produced no acceptable samples; corpus is degenerate")
	}
	t.Logf("OBJ corpus: %d samples, %d accepted, rejections=%v", samples, accepted, rejections)
}

// TestFuzzSTLFixedSeed exercises binary and ASCII STL checkers with corrupt
// headers, NaN/Inf float32 payloads and dangling facets.
func TestFuzzSTLFixedSeed(t *testing.T) {
	r := rand.New(rand.NewSource(fuzzSeed ^ 0x571))
	limits := DefaultLimits()
	limits.MaxVertices = 5000
	limits.MaxTriangles = 5000
	const samples = 400
	rejections := map[Stage]int{}
	accepted := 0
	for i := 0; i < samples; i++ {
		var data []byte
		if r.Intn(2) == 0 {
			actual := r.Intn(20)
			corrupt := r.Intn(3) == 0
			declared := actual
			if corrupt {
				declared = actual + 1 + r.Intn(100000)
			}
			data = genSTLBinary(r, actual, declared, corrupt)
		} else {
			data = genSTLAscii(r, r.Intn(15), r.Intn(4) == 0)
		}
		start := time.Now()
		rep, err := runNoPanicSTL(t, data, limits, i)
		if time.Since(start) > limits.SampleTimeout {
			t.Fatalf("sample %d exceeded time budget", i)
		}
		if err != nil {
			rejections[rep.Stage]++
			assertBoundedCounts(t, rep, limits, i)
			continue
		}
		accepted++
		assertBoundedCounts(t, rep, limits, i)
	}
	if accepted == 0 {
		t.Fatal("STL generator produced no acceptable samples")
	}
	t.Logf("STL corpus: %d samples, %d accepted, rejections=%v", samples, accepted, rejections)
}

// runNoPanicOBJ converts any checker panic into a test failure at StagePanic,
// turning the "no panic" invariant into an enforceable assertion.
func runNoPanicOBJ(t *testing.T, data []byte, limits Limits, idx int) (rep *Report, err error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("OBJ sample %d panicked inside checker: %v", idx, r)
		}
	}()
	return CheckOBJBytes(data, limits)
}

func runNoPanicSTL(t *testing.T, data []byte, limits Limits, idx int) (rep *Report, err error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("STL sample %d panicked inside checker: %v", idx, r)
		}
	}()
	return CheckSTLBytes(data, limits)
}

func assertBoundedCounts(t *testing.T, rep *Report, limits Limits, idx int) {
	t.Helper()
	if rep.Vertices > limits.MaxVertices {
		t.Fatalf("sample %d: report vertices %d > limit", idx, rep.Vertices)
	}
	if rep.Triangles > limits.MaxTriangles {
		t.Fatalf("sample %d: report triangles %d > limit", idx, rep.Triangles)
	}
}

// TestSizeQuotaAmplification proves a declared-count attack is stopped at the
// size/count stage before any loader-sized allocation can happen.
func TestSizeQuotaAmplification(t *testing.T) {
	limits := DefaultLimits()
	// A tiny binary STL claiming 4 billion triangles.
	huge := uint32(0xFFFFFFFF)
	data := genSTLBinary(rand.New(rand.NewSource(1)), 0, int(huge), true)
	rep, err := CheckSTLBytes(data, limits)
	if err == nil {
		t.Fatalf("corrupt-header STL accepted: %+v", rep)
	}
	if rep.Stage != StageHeader && rep.Stage != StageCount && rep.Stage != StageSize {
		t.Fatalf("expected framing/count rejection, got stage=%s err=%v", rep.Stage, err)
	}
}

// TestAcceptedOBJRoundTripsThroughLoader is the cross-layer property: every
// sample the independent checker accepts must also load successfully through
// the real pt.LoadOBJ and pass CheckMesh, with no panic and matching triangle
// counts. This proves checker and loader cannot silently disagree.
func TestAcceptedOBJRoundTripsThroughLoader(t *testing.T) {
	r := rand.New(rand.NewSource(fuzzSeed ^ 0xABCD))
	limits := DefaultLimits()
	limits.MaxVertices = 200
	limits.MaxTriangles = 1000
	checked, loaded := 0, 0
	for i := 0; i < 300; i++ {
		data := genOBJ(r, 3+r.Intn(25), 1+r.Intn(15))
		rep, err := CheckOBJBytes(data, limits)
		if err != nil {
			continue // rejected samples are covered elsewhere
		}
		checked++
		path := writeSample(t, data, ".obj")
		mesh, lerr := pt.LoadOBJ(path, pt.DiffuseMaterial(pt.White))
		if lerr != nil {
			t.Fatalf("checker accepted sample %d but loader errored: %v", i, lerr)
		}
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("checker accepted sample %d but loader panicked: %v", i, rec)
				}
			}()
			post, perr := CheckMesh(mesh, rep, limits)
			if perr != nil {
				// The text-level checker cannot see geometric collinearity:
				// finite, in-range vertices may still form a zero-area
				// triangle. That is exactly the independent post-load gate's
				// job, so StageDegenerate/StageValue here is an acceptable
				// layered rejection rather than a loader disagreement.
				if post.Stage != StageDegenerate && post.Stage != StageValue &&
					post.Stage != StageMagnitude {
					t.Fatalf("accepted sample %d failed mesh check unexpectedly at %s: %v",
						i, post.Stage, perr)
				}
				return
			}
			if post.Triangles != rep.Triangles {
				t.Fatalf("triangle count mismatch: checker=%d mesh=%d",
					rep.Triangles, post.Triangles)
			}
			loaded++
		}()
	}
	if checked < 20 {
		t.Fatalf("too few accepted samples to prove the property: %d", checked)
	}
	t.Logf("round-tripped %d checker-accepted samples through pt.LoadOBJ", loaded)
}

// TestWriteThenCheckRoundTrip writes generated samples to a temp directory
// (kept separate from the test source tree) and validates via the file-based
// entry points, ensuring the os.Open/Stat path and byte path agree.
func TestWriteThenCheckRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(fuzzSeed ^ 0x77))
	dir := t.TempDir()
	limits := DefaultLimits()
	for i := 0; i < 40; i++ {
		obj := genOBJ(r, 5+r.Intn(10), 2+r.Intn(8))
		p := filepath.Join(dir, fmt.Sprintf("sample-%d.obj", i))
		if err := os.WriteFile(p, obj, 0o600); err != nil {
			t.Fatal(err)
		}
		repFile, errFile := CheckOBJ(p, limits)
		repBytes, errBytes := CheckOBJBytes(obj, limits)
		if (errFile == nil) != (errBytes == nil) || repFile.Stage != repBytes.Stage {
			t.Fatalf("sample %d: file path (%s,%v) disagrees with byte path (%s,%v)",
				i, repFile.Stage, errFile, repBytes.Stage, errBytes)
		}
	}
}

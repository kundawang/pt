package safety

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"strings"
)

// Binary STL framing constants from the STL specification.
const (
	stlHeaderSize    = 84 // 80-byte header + uint32 triangle count
	stlRecordSize    = 50 // normal(12) + 3 vertices(36) + attribute(2)
	stlMaxHeaderRead = 512
)

// CheckSTL independently validates an STL file in either binary or ASCII form.
// Like CheckOBJ it never calls pt.LoadSTL: the format decision, count checks
// and finite/magnitude checks all run against the raw byte stream. A binary
// header whose declared count cannot fit in the file is rejected at
// StageHeader, which is exactly the memory-amplification pattern that makes
// loadSTLB pre-allocate a huge slice.
func CheckSTL(path string, limits Limits) (*Report, error) {
	limits = limits.fill()
	r := &Report{Format: "stl"}

	f, size, err := openLimited(path, limits, r)
	if err != nil {
		return r, err
	}
	defer f.Close()

	return checkSTL(f, size, limits, r)
}

// CheckSTLBytes is the in-memory counterpart used by the fuzzer.
func CheckSTLBytes(data []byte, limits Limits) (*Report, error) {
	limits = limits.fill()
	r := &Report{Format: "stl", Bytes: int64(len(data))}
	if int64(len(data)) > limits.MaxFileSize {
		return r.fail(StageSize, rejected(StageSize,
			"sample size %d exceeds limit %d", len(data), limits.MaxFileSize))
	}
	return checkSTL(bytes.NewReader(data), int64(len(data)), limits, r)
}

func checkSTL(src io.ReadSeeker, size int64, limits Limits, r *Report) (*Report, error) {
	r.Bytes = size
	if size < stlHeaderSize {
		// Too small to hold a binary header; only an ASCII STL can fit.
		return checkSTLAScii(src, limits, r)
	}

	// Read just enough to decide binary vs ASCII without trusting the header.
	head := make([]byte, stlMaxHeaderRead)
	n, err := io.ReadFull(src, head)
	head = head[:n]
	if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
		return r.fail(StageHeader, rejected(StageHeader, "cannot read STL header: %v", err))
	}
	if _, err := src.Seek(0, io.SeekStart); err != nil {
		return r.fail(StageHeader, rejected(StageHeader, "cannot rewind STL: %v", err))
	}

	if looksLikeASCIISTL(head) {
		r.Format = "stl-ascii"
		return checkSTLAScii(src, limits, r)
	}

	r.Format = "stl-binary"
	return checkSTLBinary(src, size, limits, r)
}

// looksLikeASCIISTL detects the textual "solid" / "facet normal" prologue.
// A binary STL can technically start with the word "solid", so the check
// requires the keyword to appear at offset 0 followed by whitespace or a
// newline and, when present, a facet/endsolid token within the prefix.
func looksLikeASCIISTL(head []byte) bool {
	trimmed := bytes.TrimLeft(head, " \t\r\n")
	if !bytes.HasPrefix(trimmed, []byte("solid")) {
		return false
	}
	rest := trimmed[len("solid"):]
	if len(rest) > 0 && rest[0] != ' ' && rest[0] != '\t' && rest[0] != '\r' && rest[0] != '\n' {
		return false
	}
	lower := bytes.ToLower(head)
	return bytes.Contains(lower, []byte("facet")) ||
		bytes.Contains(lower, []byte("endsolid")) ||
		bytes.Contains(lower, []byte("vertex"))
}

// checkSTLBinary validates framing first, then streams every triangle record.
// No slice sized by the header's declared count is ever allocated.
func checkSTLBinary(src io.Reader, size int64, limits Limits, r *Report) (*Report, error) {
	var header struct {
		_     [80]uint8
		Count uint32
	}
	if err := binary.Read(src, binary.LittleEndian, &header); err != nil {
		return r.fail(StageHeader, rejected(StageHeader, "unreadable binary STL header: %v", err))
	}
	count := int64(header.Count)
	expected := stlHeaderSize + count*stlRecordSize
	if expected != size {
		return r.fail(StageHeader, rejected(StageHeader,
			"corrupt STL header: declares %d triangles (%d bytes) but file is %d bytes",
			count, expected, size))
	}
	if count == 0 {
		return r.fail(StageDegenerate, rejected(StageDegenerate,
			"binary STL declares zero triangles"))
	}
	if int(count) > limits.MaxTriangles {
		return r.fail(StageCount, rejected(StageCount,
			"triangle count %d exceeds limit %d", count, limits.MaxTriangles))
	}

	record := make([]byte, stlRecordSize)
	for i := int64(0); i < count; i++ {
		if _, err := io.ReadFull(src, record); err != nil {
			return r.fail(StageHeader, rejected(StageHeader,
				"truncated triangle record %d: %v", i, err))
		}
		// record layout: normal [3]float32, then V1, V2, V3, then uint16 attr
		for v := 0; v < 3; v++ {
			off := 12 + v*12
			for axis := 0; axis < 3; axis++ {
				bits := binary.LittleEndian.Uint32(record[off+axis*4:])
				f := math.Float32frombits(bits)
				x := float64(f)
				if !finite(x) {
					return r.fail(StageValue, rejected(StageValue,
						"non-finite coordinate in triangle %d vertex %d", i, v))
				}
				if !bounded(x, limits.MaxCoordinate) {
					return r.fail(StageMagnitude, rejected(StageMagnitude,
						"coordinate %g in triangle %d exceeds limit %g", x, i, limits.MaxCoordinate))
				}
			}
		}
		r.Vertices += 3
		r.Triangles++
		if r.Vertices > limits.MaxVertices {
			return r.fail(StageCount, rejected(StageCount,
				"vertex count %d exceeds limit %d", r.Vertices, limits.MaxVertices))
		}
	}
	r.Faces = r.Triangles
	r.Stage = StageAccepted
	return r, nil
}

// checkSTLAScii streams "vertex x y z" records. A dangling group of one or two
// vertices (count not divisible by three) is a StageHeader framing error.
func checkSTLAScii(src io.Reader, limits Limits, r *Report) (*Report, error) {
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), maxTextLine)
	sawFacet := false
	for scanner.Scan() {
		r.Lines++
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if fields[0] == "facet" {
			sawFacet = true
		}
		if len(fields) != 4 || fields[0] != "vertex" {
			continue
		}
		for _, a := range fields[1:] {
			x, badStage, perr := parseFinite(a)
			if perr != nil {
				return r.fail(badStage, perr)
			}
			if !finite(x) {
				return r.fail(StageValue, rejected(StageValue,
					"non-finite vertex coordinate %q on line %d", a, r.Lines))
			}
			if !bounded(x, limits.MaxCoordinate) {
				return r.fail(StageMagnitude, rejected(StageMagnitude,
					"vertex coordinate %g on line %d exceeds limit %g", x, r.Lines, limits.MaxCoordinate))
			}
		}
		r.Vertices++
		if r.Vertices > limits.MaxVertices {
			return r.fail(StageCount, rejected(StageCount,
				"vertex count %d exceeds limit %d", r.Vertices, limits.MaxVertices))
		}
		if r.Vertices%3 == 0 {
			r.Triangles++
			r.Faces = r.Triangles
			if r.Triangles > limits.MaxTriangles {
				return r.fail(StageCount, rejected(StageCount,
					"triangle count %d exceeds limit %d", r.Triangles, limits.MaxTriangles))
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return r.fail(StageHeader, rejected(StageHeader, "scan error: %v", err))
	}
	if r.Vertices%3 != 0 {
		return r.fail(StageHeader, rejected(StageHeader,
			"ASCII STL ends with %d vertices not divisible by 3 (dangling facet)", r.Vertices))
	}
	if !sawFacet || r.Triangles == 0 {
		return r.fail(StageDegenerate, rejected(StageDegenerate,
			"ASCII STL contains no facet geometry (triangles=%d)", r.Triangles))
	}
	r.Stage = StageAccepted
	return r, nil
}

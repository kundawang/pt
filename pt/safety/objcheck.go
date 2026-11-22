package safety

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"strconv"
	"strings"
)

// maxTextLine caps the length of a single OBJ/ASCII-STL record. bufio.Scanner
// refuses lines longer than its buffer; we surface that as a StageHeader
// rejection instead of letting the underlying loader truncate silently.
const maxTextLine = 1 << 20 // 1 MiB per record

// CheckOBJ independently validates an OBJ file. It implements its own
// streaming parser and never invokes pt.LoadOBJ, so a defect in the renderer's
// loader cannot mask a malicious sample. The returned Report records the
// first failing stage and the resource counts observed up to that point.
func CheckOBJ(path string, limits Limits) (*Report, error) {
	limits = limits.fill()
	r := &Report{Format: "obj"}

	f, size, err := openLimited(path, limits, r)
	if err != nil {
		return r, err
	}
	defer f.Close()

	return checkOBJ(f, size, limits, r)
}

// CheckOBJBytes validates OBJ content held in memory. It exists for fuzzing:
// generated samples never have to be written through os.Create. The size
// quota is enforced against the byte slice length before any parsing begins.
func CheckOBJBytes(data []byte, limits Limits) (*Report, error) {
	limits = limits.fill()
	r := &Report{Format: "obj", Bytes: int64(len(data))}
	if int64(len(data)) > limits.MaxFileSize {
		return r.fail(StageSize, rejected(StageSize,
			"sample size %d exceeds limit %d", len(data), limits.MaxFileSize))
	}
	return checkOBJ(bytes.NewReader(data), int64(len(data)), limits, r)
}

// checkOBJ is the single shared streaming core. It reads the vertex/face
// stream itself and performs no allocation proportional to triangle count.
func checkOBJ(src io.Reader, size int64, limits Limits, r *Report) (*Report, error) {
	r.Bytes = size
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), maxTextLine)

	vertexCount := 0 // geometric "v" records; OBJ indices are 1-based
	for scanner.Scan() {
		r.Lines++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		keyword := fields[0]
		args := fields[1:]
		switch keyword {
		case "v":
			if len(args) < 3 {
				return r.fail(StageHeader, rejected(StageHeader,
					"vertex on line %d has %d coordinates, need at least 3", r.Lines, len(args)))
			}
			for _, a := range args[:3] {
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
			vertexCount++
			r.Vertices = vertexCount
			if vertexCount > limits.MaxVertices {
				return r.fail(StageCount, rejected(StageCount,
					"vertex count %d exceeds limit %d", vertexCount, limits.MaxVertices))
			}
		case "f":
			if len(args) < 3 {
				return r.fail(StageHeader, rejected(StageHeader,
					"face on line %d has %d vertices, need at least 3", r.Lines, len(args)))
			}
			// Validate every positional index; texture/normal indices are
			// ignored by pt.LoadOBJ's triangle path and need not be finite.
			seen := make(map[int]struct{}, len(args))
			for _, arg := range args {
				pos := strings.SplitN(arg, "/", 2)[0]
				n, err := strconv.Atoi(pos)
				if err != nil {
					return r.fail(StageParse, rejected(StageParse,
						"face index %q on line %d is not an integer", arg, r.Lines))
				}
				if n == 0 {
					return r.fail(StageIndex, rejected(StageIndex,
						"face index 0 on line %d is invalid (OBJ indices are 1-based)", r.Lines))
				}
				if n < 0 {
					// Negative indices are relative to the current vertex count.
					resolved := n + vertexCount
					if resolved < 1 || resolved > vertexCount {
						return r.fail(StageIndex, rejected(StageIndex,
							"relative face index %d on line %d resolves to %d outside [1,%d]",
							n, r.Lines, resolved, vertexCount))
					}
					n = resolved
				} else if n > vertexCount {
					return r.fail(StageIndex, rejected(StageIndex,
						"face index %d on line %d exceeds vertex count %d", n, r.Lines, vertexCount))
				}
				if _, dup := seen[n]; dup {
					r.warn(sprintf("line %d: face repeats vertex index %d (degenerate)", r.Lines, n))
				}
				seen[n] = struct{}{}
			}
			r.Faces++
			// An n-gon fans into n-2 triangles, matching pt.LoadOBJ.
			r.Triangles += len(args) - 2
			if r.Triangles > limits.MaxTriangles {
				return r.fail(StageCount, rejected(StageCount,
					"triangle count %d exceeds limit %d", r.Triangles, limits.MaxTriangles))
			}
		default:
			// vt, vn, mtllib, usemtl, o, g, s ... do not affect the triangle
			// vertex stream and are intentionally not counted.
		}
	}
	if err := scanner.Err(); err != nil {
		// bufio.Scanner: token too long, or an underlying read error.
		return r.fail(StageHeader, rejected(StageHeader, "scan error: %v", err))
	}
	if r.Faces == 0 || r.Triangles == 0 {
		return r.fail(StageDegenerate, rejected(StageDegenerate,
			"OBJ contains no triangulable geometry (faces=%d triangles=%d)", r.Faces, r.Triangles))
	}
	r.Stage = StageAccepted
	return r, nil
}

// openLimited opens path, records stat failures in the report and enforces the
// file-size quota before a single payload byte is parsed.
func openLimited(path string, limits Limits, r *Report) (*os.File, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		r.Stage = StageStat
		r.Err = err
		return nil, 0, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		r.Stage = StageStat
		r.Err = err
		return nil, 0, err
	}
	size := info.Size()
	if size > limits.MaxFileSize {
		f.Close()
		_, e := r.fail(StageSize, rejected(StageSize,
			"file size %d exceeds limit %d", size, limits.MaxFileSize))
		return nil, size, e
	}
	return f, size, nil
}

// degenerateArea is the cross-product area threshold used by the post-load
// geometric sanity pass (see meshcheck.go).
const degenerateArea = 1e-12

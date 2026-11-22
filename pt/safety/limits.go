// Package safety hardens the geometry input path of github.com/fogleman/pt
// against malicious or malformed OBJ, STL and SDF inputs.
//
// The reference checkers in this package parse input files independently of
// pt.LoadOBJ, pt.LoadSTL and pt.SDFShape. Every sample is bounded by file
// size, vertex/triangle counts, SDF recursion depth and wall-clock time. A
// sample must either be rejected with a structured error or produce a stable
// result; panics and silent out-of-bounds access are never allowed to escape.
package safety

import (
	"errors"
	"time"
)

// Stage identifies how far a sample got before it was rejected or accepted.
type Stage string

const (
	StageStat       Stage = "stat"       // file could not be stated / opened
	StageSize       Stage = "size"       // file exceeds MaxFileSize
	StageHeader     Stage = "header"     // malformed STL/OBJ header or framing
	StageParse      Stage = "parse"      // token / number parsing failed
	StageCount      Stage = "count"      // vertex or triangle quota exceeded
	StageIndex      Stage = "index"      // face index out of range
	StageValue      Stage = "value"      // NaN/Inf or non-finite coordinate
	StageMagnitude  Stage = "magnitude"  // coordinate exceeds MaxCoordinate
	StageDegenerate Stage = "degenerate" // zero-area / collapsed triangle
	StageDepth      Stage = "depth"      // SDF nesting exceeds MaxSDFDepth
	StageTimeout    Stage = "timeout"    // per-sample time budget exhausted
	StagePanic      Stage = "panic"      // underlying loader panicked
	StageAccepted   Stage = "accepted"   // all checks passed
)

// Limits bounds the resources a single untrusted sample may consume.
// Zero fields fall back to DefaultLimits at validation time.
type Limits struct {
	MaxFileSize   int64         // maximum accepted file size in bytes
	MaxVertices   int           // maximum geometry vertices (OBJ v / STL vertex records)
	MaxTriangles  int           // maximum triangles produced from the sample
	MaxSDFDepth   int           // maximum nested SDF evaluation depth
	MaxCoordinate float64       // maximum finite absolute coordinate magnitude
	SampleTimeout time.Duration // wall-clock budget for one validated load
}

// DefaultLimits returns conservative defaults suitable for rendering scenes.
func DefaultLimits() Limits {
	return Limits{
		MaxFileSize:   16 << 20, // 16 MiB
		MaxVertices:   1_000_000,
		MaxTriangles:  1_000_000,
		MaxSDFDepth:   256,
		MaxCoordinate: 1e6,
		SampleTimeout: 5 * time.Second,
	}
}

// fill replaces zero fields with defaults so partial Limits stay usable.
func (l Limits) fill() Limits {
	d := DefaultLimits()
	if l.MaxFileSize <= 0 {
		l.MaxFileSize = d.MaxFileSize
	}
	if l.MaxVertices <= 0 {
		l.MaxVertices = d.MaxVertices
	}
	if l.MaxTriangles <= 0 {
		l.MaxTriangles = d.MaxTriangles
	}
	if l.MaxSDFDepth <= 0 {
		l.MaxSDFDepth = d.MaxSDFDepth
	}
	if l.MaxCoordinate <= 0 {
		l.MaxCoordinate = d.MaxCoordinate
	}
	if l.SampleTimeout <= 0 {
		l.SampleTimeout = d.SampleTimeout
	}
	return l
}

// Report is the structured result of an independent reference check. It never
// depends on pt.LoadOBJ/pt.LoadSTL/pt.SDFShape internals: the checker reads
// the raw bytes itself and records where processing stopped.
type Report struct {
	Stage     Stage    // first failing stage, or StageAccepted
	Err       error    // underlying rejection reason (nil when accepted)
	Format    string   // "obj", "stl-binary" or "stl-ascii"
	Vertices  int      // geometric vertices observed up to the failing stage
	Triangles int      // triangles observed up to the failing stage
	Faces     int      // OBJ face records (STL: == Triangles)
	Bytes     int64    // file size in bytes
	Lines     int      // textual records scanned
	Warnings  []string // non-fatal anomalies (e.g. degenerate triangles)
}

// OK reports whether the sample passed every check.
func (r *Report) OK() bool { return r.Stage == StageAccepted }

func (r *Report) fail(stage Stage, err error) (*Report, error) {
	r.Stage = stage
	r.Err = err
	return r, err
}

func (r *Report) warn(msg string) {
	r.Warnings = append(r.Warnings, msg)
}

// ErrRejected wraps every structured rejection produced by the checkers so
// callers can distinguish policy failures from ordinary I/O errors.
var ErrRejected = errors.New("safety: input rejected")

func rejected(stage Stage, format string, args ...any) error {
	return &RejectionError{Stage: stage, Message: sprintf(format, args...)}
}

// RejectionError carries the failing stage alongside a human-readable message.
type RejectionError struct {
	Stage   Stage
	Message string
}

func (e *RejectionError) Error() string { return string(e.Stage) + ": " + e.Message }
func (e *RejectionError) Unwrap() error { return ErrRejected }

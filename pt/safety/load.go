package safety

import (
	"context"
	"errors"

	"github.com/fogleman/pt/pt"
)

// SafeLoadOBJ is the hardened replacement for pt.LoadOBJ on untrusted input.
//
// The pipeline is deliberately ordered so a malicious sample never reaches
// the allocation-heavy loader:
//
//  1. CheckOBJ independently parses the raw bytes (size, count, index, value,
//     magnitude and framing checks), recording the failure stage on rejection.
//  2. pt.LoadOBJ runs only after acceptance, inside a panic guard and a
//     wall-clock deadline, so an out-of-bounds index or pathological file
//     becomes a structured StagePanic/StageTimeout error instead of a crash.
//  3. CheckMesh verifies the produced mesh independently.
//
// On any failure the returned *Mesh is nil; callers never receive a partial
// mesh. The returned Report is always non-nil.
func SafeLoadOBJ(path string, material pt.Material, limits Limits) (*pt.Mesh, *Report, error) {
	limits = limits.fill()
	report, err := CheckOBJ(path, limits)
	if err != nil {
		return nil, report, err
	}

	mesh, err := guarded(func() (*pt.Mesh, error) {
		return pt.LoadOBJ(path, material)
	}, limits, report)
	if err != nil {
		return nil, report, err
	}
	return finalize(mesh, report, limits)
}

// SafeLoadSTL is the STL counterpart of SafeLoadOBJ, covering both binary and
// ASCII STL and the corrupt-header / declared-count amplification attacks.
func SafeLoadSTL(path string, material pt.Material, limits Limits) (*pt.Mesh, *Report, error) {
	limits = limits.fill()
	report, err := CheckSTL(path, limits)
	if err != nil {
		return nil, report, err
	}

	mesh, err := guarded(func() (*pt.Mesh, error) {
		return pt.LoadSTL(path, material)
	}, limits, report)
	if err != nil {
		return nil, report, err
	}
	return finalize(mesh, report, limits)
}

// guarded runs the real loader under (a) a panic-to-error boundary and (b) a
// wall-clock deadline enforced through a context. The deadline cannot kill a
// compute-bound goroutine in Go, so the panic guard covers a runaway loader
// and the context covers I/O stalls; the reference checker's count/size quotas
// are the primary bound on how much work a compliant loader can do.
func guarded(load func() (*pt.Mesh, error), limits Limits, report *Report) (*pt.Mesh, error) {
	type result struct {
		mesh *pt.Mesh
		err  error
	}
	ctx, cancel := context.WithTimeout(context.Background(), limits.SampleTimeout)
	defer cancel()

	ch := make(chan result, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				ch <- result{nil, &RejectionError{
					Stage:   StagePanic,
					Message: sprintf("underlying loader panicked: %v", r),
				}}
			}
		}()
		mesh, err := load()
		ch <- result{mesh, err}
	}()

	select {
	case res := <-ch:
		if res.err != nil {
			var rej *RejectionError
			if errors.As(res.err, &rej) {
				report.Stage = rej.Stage
			} else {
				report.Stage = StageParse
			}
			report.Err = res.err
			return nil, res.err
		}
		return res.mesh, nil
	case <-ctx.Done():
		report.Stage = StageTimeout
		report.Err = &RejectionError{
			Stage:   StageTimeout,
			Message: sprintf("loader exceeded %s budget", limits.SampleTimeout),
		}
		return nil, report.Err
	}
}

// finalize runs the post-load mesh check and normalizes any failure into the
// report.
func finalize(mesh *pt.Mesh, report *Report, limits Limits) (*pt.Mesh, *Report, error) {
	if _, err := CheckMesh(mesh, report, limits); err != nil {
		return nil, report, err
	}
	return mesh, report, nil
}

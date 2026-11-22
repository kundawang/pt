package pt

import (
	"image"
	"runtime"
)

// render keeps the historical package-level signature and renders one
// iteration into the supplied buffer. It is a thin adapter over
// Renderer so the observability counters cover both code paths.
func render(scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, buf *Buffer) {
	r := NewRenderer(scene, camera, sampler, buf.W, buf.H)
	r.Buffer = buf
	r.SamplesPerPixel = samplesPerPixel
	r.NumCPU = runtime.NumCPU()
	r.Verbose = true
	r.Run()
}

// IterativeRender renders iterations iterations into a single
// accumulating buffer, writing a PNG after every pass. It preserves
// the original package-level API used by every example.
func IterativeRender(pathTemplate string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) {
	r := NewRenderer(scene, camera, sampler, w, h)
	r.SamplesPerPixel = samplesPerPixel
	r.NumCPU = runtime.NumCPU()
	r.Verbose = true
	r.IterativeRender(pathTemplate, iterations)
}

// Render renders a single frame and returns the resulting image. It is
// a convenience wrapper around Renderer.
func Render(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) image.Image {
	r := NewRenderer(scene, camera, sampler, w, h)
	r.SamplesPerPixel = samplesPerPixel
	r.NumCPU = runtime.NumCPU()
	r.Verbose = true
	return r.Render()
}

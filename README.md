# pt: a golang path tracer

[![Build Status](https://travis-ci.org/fogleman/pt.png?branch=master)](https://travis-ci.org/fogleman/pt) [![GoDoc](https://godoc.org/github.com/fogleman/pt/pt?status.svg)](https://godoc.org/github.com/fogleman/pt/pt)

This is a CPU-only, unidirectional [path tracing](http://en.wikipedia.org/wiki/Path_tracing) engine written in Go. It has lots of features and a simple API.

![Examples](http://i.imgur.com/BTaDLC3.png)

---

### Features

* Supports OBJ and STL
* Supports textures, bump maps and normal maps
* Supports raymarching of signed distance fields
* Supports volume rendering from image slices
* Supports various material properties
* Supports configurable depth of field
* Supports iterative rendering
* Uses k-d trees to accelerate ray intersection tests
* Uses all CPU cores in parallel
* 100% pure Go with no dependencies besides the standard library

### Installation

    go get -u github.com/fogleman/pt/pt

### Examples

The are [lots of examples](https://github.com/fogleman/pt/tree/master/examples) to learn from! To try them, just run, e.g.

    cd go/src/github.com/fogleman/pt
    go run examples/gopher.go

### Optional Embree Acceleration

You can optionally utilize Intel's Embree ray tracing kernels to accelerate triangle mesh intersections. First, install embree on your system: http://embree.github.io/ Then get the `go-embree` wrapper and checkout the `embree` branch of `pt`.

    git checkout embree
    go get -u github.com/fogleman/go-embree

### Hello World

The following code demonstrates the basics of the API.

```go
package main

import . "github.com/fogleman/pt/pt"

func main() {
	// create a scene
	scene := Scene{}

	// create a material
	material := DiffuseMaterial(White)

	// add the floor (a plane)
	plane := NewPlane(V(0, 0, 0), V(0, 0, 1), material)
	scene.Add(plane)

	// add the ball (a sphere)
	sphere := NewSphere(V(0, 0, 1), 1, material)
	scene.Add(sphere)

	// add a spherical light source
	light := NewSphere(V(0, 0, 5), 1, LightMaterial(White, 8))
	scene.Add(light)

	// position the camera
	camera := LookAt(V(3, 3, 3), V(0, 0, 0.5), V(0, 0, 1), 50)

	// render the scene with progressive refinement
	sampler := NewSampler(4, 4)
	IterativeRender("out%03d.png", 1000, &scene, &camera, sampler, 960, 540, -1)
}
```

![Hello World](http://i.imgur.com/0TXY0dX.png)

### Deterministic sampling for render farms

By default `pt` uses the historical random sampler: each worker owns a
clock-seeded `math/rand` source, so the same scene renders differently on every
run and across different core counts. That is fine for a single machine, but a
render farm needs every worker to produce the same pixels.

`pt` can instead draw every random number from a scrambled, low-discrepancy
**Sobol** sequence keyed by `(seed, iteration, absolute pixel, sample ordinal,
dimension)`. The sample value is a pure function of those indices, so worker
count, scheduling, tiling and machine boundaries cannot change the image.

```go
cfg := pt.RenderConfig{
    Mode:  pt.SampleModeSobol, // deterministic, low-discrepancy
    Seed:  12345,              // same seed => same image everywhere
    // Workers and Tiles may differ per machine without affecting output.
}
img := pt.RenderWithConfig(scene, &camera, sampler, 1920, 1080, -1, cfg)
```

The zero `RenderConfig{}` keeps the legacy random behavior, so existing code is
unaffected. Low-level control is available too:

```go
seq := pt.NewSobol(12345)             // immutable; safe to share across goroutines
v := seq.Value(pixel, sample, dim)    // deterministic value in [0, 1)
r := pt.NewSobolRand(seq, pixel, 0)   // per-path cursor implementing pt.Random
r.SeekTo(pixel, sample, 0)            // O(1) jump to any point
```

Guarantees:

* **Reproducible across scheduling.** Same config + same seed yields
  bit-identical pixels for any `Workers`, `Tiles` or region split.
* **No shared mutable RNG.** `Sobol` is immutable; each path gets its own
  `SobolRand` cursor.
* **Defined edge cases.** Negative pixel/sample indices clamp to 0; `Skip` with
  a non-positive count and `Intn(n <= 0)` are defined no-ops; non-finite seeds
  are rejected by `NewSobolFromSeed`; dimensions past the 1024 available Sobol
  dimensions switch to a documented deterministic hash rather than panicking.
* **Auditable order.** `Sobol.TraceSamples` and a golden render-hash test pin
  the exact sampling order, so changes are explicit.

#### Splitting one image across machines

`cmd/farm` renders regions independently and merges them. Because samples key
on absolute pixel coordinates, the merged image is identical to rendering the
whole frame on one machine.

```
farm plan   -scene spheres -w 1920 -h 1080 -shards 16
farm render -scene spheres -w 1920 -h 1080 -shards 16 -shard 3 -seed 12345 -out shard3.png
farm merge  -out final.png shard0.png shard1.png ...
```

Each shard gets a `.json` sidecar recording its region and sampling parameters;
`farm merge` rejects mismatched seeds, overlapping regions or uncovered pixels
instead of silently producing a corrupt image.

### Links

Here are some resources that I have found useful.

* [WebGL Path Tracing - Evan Wallace](http://madebyevan.com/webgl-path-tracing/)
* [Global Illumination in a Nutshell](http://www.thepolygoners.com/tutorials/GIIntro/GIIntro.htm)
* [Simple Path Tracing - Iñigo Quilez](http://www.iquilezles.org/www/articles/simplepathtracing/simplepathtracing.htm)
* [Realistic Raytracing - Zack Waters](http://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/realistic_raytracing.html)
* [Reflections and Refractions in Ray Tracing - Bram de Greve](http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf)
* [Better Sampling - Rory Driscoll](http://www.rorydriscoll.com/2009/01/07/better-sampling/)
* [Ray Tracing for Global Illumination - Nelson Max at UC Davis](https://www.youtube.com/playlist?list=PLslgisHe5tBPckSYyKoU3jEA4bqiFmNBJ)
* [Physically Based Rendering - Matt Pharr, Greg Humphreys](http://www.amazon.com/Physically-Based-Rendering-Second-Edition/dp/0123750792)

### Samples

![Go](http://i.imgur.com/LMNUoaM.jpg)

![Dragon](https://www.michaelfogleman.com/static/gallery/out1000c.png)

![Cornell](https://www.michaelfogleman.com/static/gallery/853.png)

![Lucy](https://www.michaelfogleman.com/static/gallery/756b.png)

![SDF](https://www.michaelfogleman.com/static/gallery/470d.png)

![Spheres](https://www.michaelfogleman.com/static/gallery/dof.png)

![Suzanne](http://i.imgur.com/iw32US1.png)

![Molecule](https://www.michaelfogleman.com/static/gallery/600d.png)

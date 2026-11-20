package main

import . "github.com/fogleman/pt/pt"

// Motion blur demo: a glowing sphere sweeps horizontally across the frame
// while the shutter is open, leaving a streak. A static control sphere is
// rendered for comparison. Run with: go run examples/motionblur.go
func main() {
	// moving sphere: sweeps from x=-2 to x=2 during shutter interval [0, 1]
	scene := Scene{}
	material := LightMaterial(White, 5)
	scene.Add(NewMovingSphere(V(-2, 0, 0), V(2, 0, 0), 0.5, material, 0, 1))
	camera := LookAt(V(0, 0, 6), V(0, 0, 0), V(0, 1, 0), 40)
	camera.SetShutter(0, 1)
	sampler := NewDirectSampler()
	renderer := NewRenderer(&scene, &camera, sampler, 480, 320)
	renderer.SamplesPerPixel = 256
	renderer.Seed = 1
	renderer.IterativeRender("motionblur.png", 1)

	// static control: same sphere frozen at the path midpoint
	scene2 := Scene{}
	scene2.Add(NewSphere(V(0, 0, 0), 0.5, material))
	camera2 := LookAt(V(0, 0, 6), V(0, 0, 0), V(0, 1, 0), 40)
	renderer2 := NewRenderer(&scene2, &camera2, sampler, 480, 320)
	renderer2.SamplesPerPixel = 256
	renderer2.Seed = 1
	renderer2.IterativeRender("motionblur-static.png", 1)
}

package main

import (
	"fmt"

	"github.com/fogleman/pt/pt"
)

// buildScene constructs the named scene. Every worker must build the same
// scene, so all geometry, materials and camera parameters live here rather
// than in per-machine flags. Add new scenes in this one switch.
func buildScene(name string, spp int) (*pt.Scene, *pt.Camera, pt.Sampler, error) {
	switch name {
	case "cornell":
		return cornellScene(spp)
	case "spheres":
		return spheresScene(spp)
	default:
		return nil, nil, nil, fmt.Errorf("unknown scene %q (want cornell or spheres)", name)
	}
}

// cornellScene loads the Cornell box that ships with the examples and uses
// the camera and sampler from examples/cornell.go.
func cornellScene(spp int) (*pt.Scene, *pt.Camera, pt.Sampler, error) {
	scene := &pt.Scene{}
	material := pt.DiffuseMaterial(pt.Black)
	mesh, err := pt.LoadOBJ("examples/CornellBox-Original.obj", material)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load CornellBox-Original.obj: %w", err)
	}
	for _, t := range mesh.Triangles {
		scene.Add(t)
	}
	camera := pt.LookAt(pt.V(0, 1, 3), pt.V(0, 1, 0), pt.V(0, 1, 0), 50)
	sampler := pt.NewSampler(4, 8)
	if spp < 0 {
		spp = 4
	}
	return scene, &camera, sampler, nil
}

// spheresScene is a small procedural scene with a floor, three spheres and a
// light, useful when the Cornell OBJ is not available.
func spheresScene(spp int) (*pt.Scene, *pt.Camera, pt.Sampler, error) {
	scene := &pt.Scene{Color: pt.Color{R: 0.1, G: 0.1, B: 0.12}}
	floor := pt.DiffuseMaterial(pt.Color{R: 0.8, G: 0.8, B: 0.8})
	scene.Add(pt.NewPlane(pt.V(0, 0, 0), pt.V(0, 0, 1), floor))
	scene.Add(pt.NewSphere(pt.V(-1.1, 0, 0.9), 0.9, pt.DiffuseMaterial(pt.Color{R: 0.8, G: 0.2, B: 0.2})))
	scene.Add(pt.NewSphere(pt.V(0, 0, 0.75), 0.75, pt.GlossyMaterial(pt.Color{R: 0.8, G: 0.8, B: 0.8}, 1.5, 0.1)))
	scene.Add(pt.NewSphere(pt.V(1.1, 0, 0.85), 0.85, pt.DiffuseMaterial(pt.Color{R: 0.2, G: 0.3, B: 0.8})))
	scene.Add(pt.NewSphere(pt.V(0, 0, 4), 0.5, pt.LightMaterial(pt.White, 25)))
	camera := pt.LookAt(pt.V(0, -3, 2.2), pt.V(0, 0, 0.9), pt.V(0, 0, 1), 50)
	sampler := pt.NewSampler(4, 6)
	_ = spp
	return scene, &camera, sampler, nil
}

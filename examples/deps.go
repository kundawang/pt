//go:build tools

// This file pins the external dependencies used only by the
// //go:build ignore-tagged example programs in this directory
// (counties.go uses go-colorful, mol.go uses mol,
// noise.go uses opensimplex-go), so that
// "go mod tidy" keeps them in go.mod and the examples stay
// runnable via e.g. "go run examples/counties.go".
//
// It is never compiled: the "tools" build tag is never set.
package main

import (
	_ "github.com/fogleman/mol/mol"
	_ "github.com/lucasb-eyer/go-colorful"
	_ "github.com/ojrac/opensimplex-go"
)

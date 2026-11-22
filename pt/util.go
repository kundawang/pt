package pt

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"path"
	"strconv"
	"time"
)

func Radians(degrees float64) float64 {
	return degrees * math.Pi / 180
}

func Degrees(radians float64) float64 {
	return radians * 180 / math.Pi
}

// OrthonormalBasis returns two unit vectors perpendicular to n and to each
// other. It uses Duff et al.'s branchless construction, which is continuous in
// n and consumes no randomness. Deterministic samplers must use this rather
// than a rejection-sampled vector: consuming a variable number of dimensions
// per path would desynchronize the low-discrepancy sequence and give up its
// stratification.
func OrthonormalBasis(n Vector) (Vector, Vector) {
	sign := 1.0
	if n.Z < 0 {
		sign = -1.0
	}
	a := -1.0 / (sign + n.Z)
	b := n.X * n.Y * a
	s := Vector{1 + sign*n.X*n.X*a, sign * b, -sign * n.X}
	t := Vector{b, sign + n.Y*n.Y*a, -n.Y}
	return s, t
}

// ConcentricDisk maps a point (u, v) in [0,1)^2 to a point inside the unit
// disk using Shirley's concentric mapping. Unlike rejection sampling it always
// consumes exactly two values, which deterministic samplers require.
func ConcentricDisk(u, v float64) (float64, float64) {
	a := 2*u - 1
	b := 2*v - 1
	if a == 0 && b == 0 {
		return 0, 0
	}
	var r, phi float64
	if math.Abs(a) > math.Abs(b) {
		r = a
		phi = (math.Pi / 4) * (b / a)
	} else {
		r = b
		phi = (math.Pi / 2) - (math.Pi/4)*(a/b)
	}
	return r * math.Cos(phi), r * math.Sin(phi)
}

// coneBasis returns the tangent basis used to build a cone sample. SobolRand
// selects the deterministic basis; every other Random (including the legacy
// random path) keeps the historical rejection-sampled basis.
func coneBasis(direction Vector, rnd Random) (Vector, Vector) {
	if _, ok := rnd.(*SobolRand); ok {
		return OrthonormalBasis(direction)
	}
	q := RandomUnitVector(rnd)
	s := direction.Cross(q)
	return s, direction.Cross(s)
}

func Cone(direction Vector, theta, u, v float64, rnd Random) Vector {
	if theta < EPS {
		return direction
	}
	theta = theta * (1 - (2 * math.Acos(u) / math.Pi))
	m1 := math.Sin(theta)
	m2 := math.Cos(theta)
	a := v * 2 * math.Pi
	s, t := coneBasis(direction, rnd)
	d := Vector{}
	d = d.Add(s.MulScalar(m1 * math.Cos(a)))
	d = d.Add(t.MulScalar(m1 * math.Sin(a)))
	d = d.Add(direction.MulScalar(m2))
	d = d.Normalize()
	return d
}

func LoadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	return im, err
}

func SavePNG(path string, im image.Image) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return png.Encode(file, im)
}

func Median(items []float64) float64 {
	n := len(items)
	switch {
	case n == 0:
		return 0
	case n%2 == 1:
		return items[n/2]
	default:
		a := items[n/2-1]
		b := items[n/2]
		return (a + b) / 2
	}
}

func DurationString(d time.Duration) string {
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60
	return fmt.Sprintf("%d:%02d:%02d", h, m, s)
}

func NumberString(x float64) string {
	suffixes := []string{"", "k", "M", "G"}
	for _, suffix := range suffixes {
		if x < 1000 {
			return fmt.Sprintf("%.1f%s", x, suffix)
		}
		x /= 1000
	}
	return fmt.Sprintf("%.1f%s", x, "T")
}

func ParseFloats(items []string) []float64 {
	result := make([]float64, len(items))
	for i, item := range items {
		f, _ := strconv.ParseFloat(item, 64)
		result[i] = f
	}
	return result
}

func ParseInts(items []string) []int {
	result := make([]int, len(items))
	for i, item := range items {
		f, _ := strconv.ParseInt(item, 0, 0)
		result[i] = int(f)
	}
	return result
}

func RelativePath(path1, path2 string) string {
	dir, _ := path.Split(path1)
	return path.Join(dir, path2)
}

func Fract(x float64) float64 {
	_, x = math.Modf(x)
	return x
}

func Clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func ClampInt(x, lo, hi int) int {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

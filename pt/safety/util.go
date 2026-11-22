package safety

import (
	"fmt"
	"math"
	"strconv"
)

func sprintf(format string, args ...any) string { return fmt.Sprintf(format, args...) }

// parseFinite parses a float64 token and classifies the outcome into a
// failure stage. An out-of-range token such as "1e999" is clamped to ±Inf by
// strconv with ErrRange: that is a non-finite *value* defect (StageValue),
// not a malformed token (StageParse). It returns the parsed value, the stage
// (empty on success) and a non-nil error on failure.
func parseFinite(token string) (float64, Stage, error) {
	x, err := strconv.ParseFloat(token, 64)
	if err != nil {
		if ne, ok := err.(*strconv.NumError); ok && ne.Err == strconv.ErrRange {
			return x, StageValue, rejected(StageValue, "coordinate %q overflows float64 (%g)", token, x)
		}
		return 0, StageParse, rejected(StageParse, "coordinate %q is not a float64", token)
	}
	return x, "", nil
}

// finite reports whether x is a usable real number. NaN and Inf are rejected
// outright: once they enter a Vector they poison k-d tree splits, ray tests
// and bounding boxes with silent, non-deterministic geometry.
func finite(x float64) bool {
	// NaN is the only value not equal to itself; Inf fails the Inf check.
	return x == x && !isInf(x)
}

func isInf(x float64) bool {
	// avoid importing math in hot paths via a constant-free test
	return math.IsInf(x, 0)
}

// bounded reports whether |x| is within the coordinate magnitude budget.
func bounded(x, max float64) bool {
	if x < 0 {
		x = -x
	}
	return x <= max
}

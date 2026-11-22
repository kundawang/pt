package pt

type Material struct {
	Color          Color
	Texture        Texture
	NormalTexture  Texture
	BumpTexture    Texture
	GlossTexture   Texture
	BumpMultiplier float64
	Emittance      float64
	Index          float64 // refractive index
	Gloss          float64 // reflection cone angle in radians
	Tint           float64 // specular and refractive tinting
	Reflectivity   float64 // metallic reflection
	Transparent    bool
}

func DiffuseMaterial(color Color) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, 1, 0, 0, -1, false}
}

func SpecularMaterial(color Color, index float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, index, 0, 0, -1, false}
}

func GlossyMaterial(color Color, index, gloss float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, index, gloss, 0, -1, false}
}

func ClearMaterial(index, gloss float64) Material {
	return Material{Black, nil, nil, nil, nil, 1, 0, index, gloss, 0, -1, true}
}

func TransparentMaterial(color Color, index, gloss, tint float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, index, gloss, tint, -1, true}
}

func MetallicMaterial(color Color, gloss, tint float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, 1, gloss, tint, 1, false}
}

func LightMaterial(color Color, emittance float64) Material {
	return Material{color, nil, nil, nil, nil, 1, emittance, 1, 0, 0, -1, false}
}

// VertexColoredShape is implemented by shapes that can supply a color
// interpolated from per-vertex colors at the given point.
type VertexColoredShape interface {
	VertexColorAt(p Vector) Color
	HasVertexColors() bool
}

// MaterialAt resolves the material seen at a hit point. Composition order,
// from first to last:
//
//  1. The shape's base Material (constant color, NormalTexture/BumpTexture,
//     Emittance, Transparent and other flags are taken as-is).
//  2. If the shape carries per-vertex colors, the barycentric-interpolated
//     vertex color replaces the material's constant Color.
//  3. If the material also has a color Texture, the texture sample
//     modulates (multiplies) the vertex color; without vertex colors the
//     texture replaces the color exactly as before.
//  4. GlossTexture (and the not-listed NormalTexture/BumpTexture) is
//     independent of color and is handled unchanged.
//
// Emittance and transparency therefore keep working unchanged: a vertex
// colored light emits Emittance * vertex color, and a transparent vertex
// colored material tints refraction with the interpolated vertex color.
// Shapes without vertex colors keep their original, bit-for-bit behavior.
func MaterialAt(shape Shape, point Vector) Material {
	material := shape.MaterialAt(point)
	uv := shape.UV(point)
	if colored, ok := shape.(VertexColoredShape); ok && colored.HasVertexColors() {
		material.Color = colored.VertexColorAt(point)
	}
	if material.Texture != nil {
		textureColor := material.Texture.Sample(uv.X, uv.Y)
		if colored, ok := shape.(VertexColoredShape); ok && colored.HasVertexColors() {
			material.Color = material.Color.Mul(textureColor)
		} else {
			material.Color = textureColor
		}
	}
	if material.GlossTexture != nil {
		c := material.GlossTexture.Sample(uv.X, uv.Y)
		material.Gloss = (c.R + c.G + c.B) / 3
	}
	return material
}

import { NextResponse } from "next/server"

export async function POST() {
  try {
    // Appeler le script Python pour entraîner le modèle
    const response = await fetch("http://localhost:8000/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error("Erreur lors de l'entraînement du modèle")
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Erreur d'entraînement:", error)
    return NextResponse.json({ error: "Erreur lors de l'entraînement du modèle" }, { status: 500 })
  }
}

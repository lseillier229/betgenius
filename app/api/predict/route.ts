import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { red_fighter, blue_fighter } = body

    if (!red_fighter || !blue_fighter) {
      return NextResponse.json({ error: "Les deux combattants doivent être spécifiés" }, { status: 400 })
    }

    // Appeler le script Python pour faire la prédiction
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        red_fighter,
        blue_fighter,
      }),
    })

    if (!response.ok) {
      throw new Error("Erreur lors de la prédiction")
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Erreur de prédiction:", error)
    return NextResponse.json({ error: "Erreur lors de la prédiction" }, { status: 500 })
  }
}

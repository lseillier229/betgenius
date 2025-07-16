import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Simuler une liste de combattants (en production, cela viendrait de votre dataset)
    const fighters = [
      "Jon Jones",
      "Daniel Cormier",
      "Stipe Miocic",
      "Francis Ngannou",
      "Conor McGregor",
      "Khabib Nurmagomedov",
      "Tony Ferguson",
      "Dustin Poirier",
      "Israel Adesanya",
      "Robert Whittaker",
      "Yoel Romero",
      "Paulo Costa",
      "Kamaru Usman",
      "Colby Covington",
      "Jorge Masvidal",
      "Leon Edwards",
      "Max Holloway",
      "Alexander Volkanovski",
      "Jose Aldo",
      "Brian Ortega",
      "Henry Cejudo",
      "Aljamain Sterling",
      "Petr Yan",
      "Cory Sandhagen",
      "Deiveson Figueiredo",
      "Brandon Moreno",
      "Askar Askarov",
      "Alexandre Pantoja",
    ].sort()

    return NextResponse.json({ fighters })
  } catch (error) {
    return NextResponse.json({ error: "Erreur lors du chargement des combattants" }, { status: 500 })
  }
}

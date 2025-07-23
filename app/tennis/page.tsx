"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Loader2, Trophy, Users, Target, ArrowLeft, ArrowRight } from "lucide-react"
import Link from "next/link"

interface TennisPlayer {
  name: string
  matches_count: number
  avg_rank: number | null
}

interface TennisPrediction {
  player1: string
  player2: string
  surface: string
  tourney_level: string
  player1_probability: number
  player2_probability: number
  predicted_winner: string
  confidence: number
}

const TENNIS_API = "http://localhost:8000"

export default function TennisPage() {
  const [players, setPlayers] = useState<TennisPlayer[]>([])
  const [surfaces, setSurfaces] = useState<string[]>([])
  const [tourneyLevels, setTourneyLevels] = useState<string[]>([])
  const [player1, setPlayer1] = useState("")
  const [player2, setPlayer2] = useState("")
  const [surface, setSurface] = useState("Hard")
  const [tourneyLevel, setTourneyLevel] = useState("A")
  const [modelTrained, setModelTrained] = useState(false)
  const [trainingModel, setTrainingModel] = useState(false)
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState<TennisPrediction | null>(null)

  useEffect(() => {
    loadPlayers()
    loadSurfaces()
    checkModelStatus()
  }, [])

  const loadPlayers = async () => {
    try {
      const response = await fetch(`${TENNIS_API}/tennis/players`)
      const data = await response.json()
      setPlayers(data.players)
    } catch (err) {
      console.error("Erreur chargement joueurs:", err)
    }
  }

  const loadSurfaces = async () => {
    try {
      const response = await fetch(`${TENNIS_API}/tennis/surfaces`)
      const data = await response.json()
      setSurfaces(data.surfaces)
      setTourneyLevels(data.tourney_levels)
    } catch (err) {
      console.error("Erreur chargement surfaces:", err)
    }
  }

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${TENNIS_API}/tennis/health`)
      const data = await response.json()
      setModelTrained(data.model_trained)
    } catch (err) {
      console.error("Erreur status modèle:", err)
    }
  }

  const trainModel = async () => {
    setTrainingModel(true)
    try {
      const response = await fetch(`${TENNIS_API}/tennis/train`, { method: "POST" })
      const data = await response.json()
      if (data.success) {
        setModelTrained(true)
        console.log(`Modèle entraîné avec accuracy: ${data.accuracy}`)
      }
    } catch (err) {
      console.error("Erreur entraînement:", err)
    } finally {
      setTrainingModel(false)
    }
  }

  const makePrediction = async () => {
    if (!player1 || !player2) return
    setLoading(true)
    try {
      const response = await fetch(`${TENNIS_API}/tennis/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player1,
          player2,
          surface,
          tourney_level: tourneyLevel
        })
      })
      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      console.error("Erreur prédiction:", err)
    } finally {
      setLoading(false)
    }
  }

  const getSurfaceEmoji = (surf: string) => {
    switch (surf) {
      case "Hard": return "🏟️"
      case "Clay": return "🧱"
      case "Grass": return "🌱"
      case "Carpet": return "🏢"
      default: return "🎾"
    }
  }

  const getTourneyLevelName = (level: string) => {
    switch (level) {
      case "G": return "Grand Slam"
      case "M": return "Masters"
      case "A": return "ATP"
      case "D": return "Davis Cup"
      default: return level
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* Navigation */}
        <div className="flex items-center gap-4">
          <Link href="/">
            <Button variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Retour UFC
            </Button>
          </Link>
        </div>

        {/* Titre */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold flex items-center justify-center gap-2">
            <Trophy className="h-8 w-8 text-yellow-500" /> BETGENIUS TENNIS
          </h1>
          <p className="text-gray-600">Prédisez le vainqueur de vos matchs tennis favoris</p>
        </div>

        {/* Entraînement du modèle */}
        {/*!modelTrained && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" /> Entraînement du modèle
              </CardTitle>
              <CardDescription>Le modèle tennis doit être entraîné avant les prédictions</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={trainModel} disabled={trainingModel} className="w-full">
                {trainingModel
                  ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Entraînement…</>)
                  : "Entraîner le modèle tennis"}
              </Button>
            </CardContent>
          </Card>
        )*/}

        {/* Sélection des joueurs et paramètres */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" /> Configuration du match
            </CardTitle>
            <CardDescription>Choisissez les joueurs et les conditions du match</CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* Joueurs */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-blue-600">🎾 Joueur 1</label>
                <Select value={player1} onValueChange={setPlayer1}>
                  <SelectTrigger className="border-blue-200">
                    <SelectValue placeholder="Sélectionner joueur 1" />
                  </SelectTrigger>
                  <SelectContent>
                    {players.map(p => (
                      <SelectItem key={p.name} value={p.name}>
                        <div className="flex items-center justify-between w-full">
                          <span>{p.name}</span>
                          <span className="text-xs text-gray-500 ml-2">

                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-red-600">🎾 Joueur 2</label>
                <Select value={player2} onValueChange={setPlayer2}>
                  <SelectTrigger className="border-red-200">
                    <SelectValue placeholder="Sélectionner joueur 2" />
                  </SelectTrigger>
                  <SelectContent>
                    {players.map(p => (
                      <SelectItem key={p.name} value={p.name}>
                        <div className="flex items-center justify-between w-full">
                          <span>{p.name}</span>
                          <span className="text-xs text-gray-500 ml-2">
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Conditions du match */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Surface</label>
                <Select value={surface} onValueChange={setSurface}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {surfaces.map(s => (
                      <SelectItem key={s} value={s}>
                        {getSurfaceEmoji(s)} {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Niveau de tournoi</label>
                <Select value={tourneyLevel} onValueChange={setTourneyLevel}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {tourneyLevels.map(level => (
                      <SelectItem key={level} value={level}>
                        {getTourneyLevelName(level)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Button
              onClick={makePrediction}
              disabled={!player1 || !player2  || loading}
              className="w-full"
            >
              {loading
                ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Prédiction…</>)
                : "Prédire le vainqueur"}
            </Button>
          </CardContent>
        </Card>

        {/* Résultat */}
        {prediction && (
          <Card>
            <CardHeader>
              <CardTitle>🎯 Résultat de la prédiction</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              
              {/* Info du match */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex justify-between items-center text-sm text-gray-600">
                  <span>Surface: {getSurfaceEmoji(prediction.surface)} {prediction.surface}</span>
                  <span>Niveau: {getTourneyLevelName(prediction.tourney_level)}</span>
                </div>
              </div>

              {/* Vainqueur prédit */}
              <div className="text-center">
                <Badge
                  variant={prediction.predicted_winner === prediction.player1 ? "default" : "secondary"}
                  className="text-lg px-4 py-2"
                >
                  Vainqueur prédit : {prediction.predicted_winner}
                </Badge>
                <p className="text-sm text-gray-600 mt-2">
                  Confiance : {(prediction.confidence * 100).toFixed(1)}%
                </p>
              </div>

              {/* Probabilités */}
              <div className="space-y-4">
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="font-medium text-blue-600">🎾 {prediction.player1}</span>
                    <span className="font-bold">{(prediction.player1_probability * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prediction.player1_probability * 100} className="h-3" />
                </div>
                
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="font-medium text-red-600">🎾 {prediction.player2}</span>
                    <span className="font-bold">{(prediction.player2_probability * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prediction.player2_probability * 100} className="h-3" />
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
"use client"

import { useEffect, useState } from "react"
import Image from "next/image"

import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

import { Loader2, Trophy, Users, Target } from "lucide-react"

/* ───────── types ───────── */
interface Fighter {
  name: string
  has_img: boolean
  img: string | null
}

interface PredictionResult {
  red_fighter: string
  blue_fighter: string
  red_probability: number
  blue_probability: number
  predicted_winner: string
  confidence: number
}

/* ───────── config ───────── */
const API      = "http://localhost:8000"          // backend Flask
const FALLBACK = "/images/default-img.png"            // image statique dans /public

export default function Home() {
  /* ───────── états ───────── */
  const [fighters,      setFighters]      = useState<Fighter[]>([])
  const [redFighter,    setRedFighter]    = useState("")
  const [blueFighter,   setBlueFighter]   = useState("")
  const [modelTrained,  setModelTrained]  = useState(false)
  const [trainingModel, setTrainingModel] = useState(false)
  const [loading,       setLoading]       = useState(false)
  const [prediction,    setPrediction]    = useState<PredictionResult | null>(null)

  /* ───────── chargement des combattants ───────── */
  useEffect(() => {
    fetch(`${API}/fighters`)
      .then(r => r.json())
      .then(d => setFighters(d.fighters as Fighter[]))
      .catch(err => console.error("load fighters:", err))
  }, [])

  /* ───────── entraînement ───────── */
  const trainModel = async () => {
    setTrainingModel(true)
    try {
      const r = await fetch(`${API}/train`, { method: "POST" })
      const d = await r.json()
      if (d.success) setModelTrained(true)
    } catch (err) {
      console.error("train error:", err)
    } finally {
      setTrainingModel(false)
    }
  }

  /* ───────── prédiction ───────── */
  const makePrediction = async () => {
    if (!redFighter || !blueFighter) return
    setLoading(true)
    try {
      const r = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ red_fighter: redFighter, blue_fighter: blueFighter })
      })
      const d = await r.json()
      setPrediction(d)
    } catch (err) {
      console.error("predict error:", err)
    } finally {
      setLoading(false)
    }
  }

  /* ───────── rendu ───────── */
  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* titre */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold flex items-center justify-center gap-2">
            <Trophy className="h-8 w-8 text-yellow-500" /> BETGENIUS
          </h1>
          <p className="text-gray-600">Prédisez le vainqueur de vos combats UFC favoris</p>
        </div>

        {/* bloc entraînement */}
        {!modelTrained && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" /> Entraînement du modèle
              </CardTitle>
              <CardDescription>Le modèle doit être entraîné avant les prédictions</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={trainModel} disabled={trainingModel} className="w-full">
                {trainingModel
                  ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Entraînement…</>)
                  : "Entraîner le modèle"}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* sélection combattants */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" /> Sélection des combattants
            </CardTitle>
            <CardDescription>Choisissez les deux combattants</CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              {/* red */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-red-600">👊 Red Corner</label>
                <Select value={redFighter} onValueChange={setRedFighter}>
                  <SelectTrigger className="border-red-200">
                    <SelectValue placeholder="Combattant rouge" />
                  </SelectTrigger>
                  <SelectContent>
                    {fighters.map(f => {
                      // ── choisis la bonne source ─────────────────
                      const src = f.has_img && f.img ? f.img : FALLBACK
                      return (
                        <SelectItem key={f.name} value={f.name}>
                          <div className="flex items-center gap-2">
                            <Image
                              src={src}                     /* ← ici */
                              alt={f.name}
                              width={24}
                              height={24}
                              className="rounded-full border"
                              unoptimized
                            />
                            <span>{f.name}</span>
                          </div>
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>
              </div>

              {/* blue */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-blue-600">💥 Blue Corner</label>
                <Select value={blueFighter} onValueChange={setBlueFighter}>
                  <SelectTrigger className="border-blue-200">
                    <SelectValue placeholder="Combattant bleu" />
                  </SelectTrigger>
                  <SelectContent>
                    {fighters.map(f => {
                      // ── choisis la bonne source ─────────────────
                      const src = f.has_img && f.img ? f.img : FALLBACK
                      return (
                        <SelectItem key={f.name} value={f.name}>
                          <div className="flex items-center gap-2">
                            <Image
                              src={src}                     /* ← ici */
                              alt={f.name}
                              width={24}
                              height={24}
                              className="rounded-full border"
                              unoptimized
                            />
                            <span>{f.name}</span>
                          </div>
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Button
              onClick={makePrediction}
              disabled={!redFighter || !blueFighter || !modelTrained || loading}
              className="w-full"
            >
              {loading
                ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Prédiction…</>)
                : "Prédire le vainqueur"}
            </Button>
          </CardContent>
        </Card>

        {/* résultat */}
        {prediction && (
          <Card>
            <CardHeader><CardTitle>🎯 Résultat</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center">
                <Badge
                  variant={prediction.predicted_winner === redFighter ? "destructive" : "default"}
                  className="text-lg px-4 py-2"
                >
                  Vainqueur prédit&nbsp;: {prediction.predicted_winner}
                </Badge>
                <p className="text-sm text-gray-600 mt-2">
                  Confiance&nbsp;: {(prediction.confidence * 100).toFixed(1)}%
                </p>
              </div>

              <div className="space-y-4">
                {/* red */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="font-medium text-red-600">🔴 {redFighter}</span>
                    <span className="font-bold">{(prediction.red_probability * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prediction.red_probability * 100} className="h-3" />
                </div>
                {/* blue */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="font-medium text-blue-600">🔵 {blueFighter}</span>
                    <span className="font-bold">{(prediction.blue_probability * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prediction.blue_probability * 100} className="h-3" />
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

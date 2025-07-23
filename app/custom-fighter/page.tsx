"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Loader2, UserPlus, Trash2, ArrowLeft } from "lucide-react"
import { motion } from "framer-motion"

interface Fighter {
  name: string
  age: number
  height: number
  weight: number
  reach?: number
  stance?: string
  weight_class: string
  gender: string
  created_at?: string
}

const API = "http://localhost:8000"

const weightClasses = [
  "Flyweight",
  "Bantamweight",
  "Featherweight",
  "Lightweight",
  "Welterweight",
  "Middleweight",
  "Light Heavyweight",
  "Heavyweight",
]

const genders = ["Male", "Female"]
const stances = ["Orthodox", "Southpaw", "Switch", "Unknown"]

export default function CustomFighterPage() {
  /** Form state */
  const [form, setForm] = useState<Fighter>({
    name: "",
    age: 0,
    height: 0,
    weight: 0,
    reach: undefined,
    stance: "Orthodox",
    weight_class: "Lightweight",
    gender: "Male",
  })

  /** List state */
  const [fighters, setFighters] = useState<Fighter[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [loading, setLoading] = useState(true)

  /** Fetch fighters on mount */
  useEffect(() => {
    fetchFighters()
  }, [])

  const fetchFighters = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/custom-fighters`)
      const d = await r.json()
      setFighters(d.fighters as Fighter[])
    } catch (e) {
      console.error("load custom fighters:", e)
    } finally {
      setLoading(false)
    }
  }

  /** Update form helper */
  const updateForm = (key: keyof Fighter, value: string | number | undefined) => {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  /** Submit new fighter */
  const submitFighter = async () => {
    if (!form.name || !form.age || !form.height || !form.weight) return
    setSubmitting(true)
    try {
      const payload = {
        ...form,
        age: Number(form.age),
        height: Number(form.height),
        weight: Number(form.weight),
        reach: form.reach ? Number(form.reach) : undefined,
      }
      const r = await fetch(`${API}/custom-fighter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const d = await r.json()
      if (r.ok && d.success) {
        setForm({
          name: "",
          age: 0,
          height: 0,
          weight: 0,
          reach: undefined,
          stance: "Orthodox",
          weight_class: "Lightweight",
          gender: "Male",
        })
        fetchFighters()
      } else {
        alert(d.error || "Erreur inconnue")
      }
    } catch (e) {
      console.error("add fighter error:", e)
    } finally {
      setSubmitting(false)
    }
  }

  /** Delete fighter */
  const deleteFighter = async (name: string) => {
    if (!confirm(`Supprimer ${name} ?`)) return
    try {
      await fetch(`${API}/custom-fighter/${encodeURIComponent(name)}`, {
        method: "DELETE",
      })
      fetchFighters()
    } catch (e) {
      console.error("delete error:", e)
    }
  }

  return (
    <motion.div
      className="min-h-screen bg-gradient-to-br from-red-50 to-blue-50 p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-2">
          <Link href="/">
            <Button variant="ghost" size="icon" className="rounded-full">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <h1 className="text-3xl font-bold">Mes combattants personnalisés</h1>
        </div>

        {/* Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserPlus className="h-5 w-5" /> Ajouter un combattant
            </CardTitle>
            <CardDescription>Complétez le formulaire puis validez</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <Input
                placeholder="Nom"
                value={form.name}
                onChange={e => updateForm("name", e.target.value)}
              />
              <Input
                placeholder="Âge"
                type="number"
                min={0}
                value={form.age || ""}
                onChange={e => updateForm("age", Number(e.target.value))}
              />
              <Input
                placeholder="Taille (cm)"
                type="number"
                min={0}
                value={form.height || ""}
                onChange={e => updateForm("height", Number(e.target.value))}
              />
              <Input
                placeholder="Poids (kg)"
                type="number"
                min={0}
                value={form.weight || ""}
                onChange={e => updateForm("weight", Number(e.target.value))}
              />
              <Input
                placeholder="Allonge (cm, optionnel)"
                type="number"
                min={0}
                value={form.reach || ""}
                onChange={e => updateForm("reach", e.target.value ? Number(e.target.value) : undefined)}
              />
              {/* Weight class */}
              <Select
                value={form.weight_class}
                onValueChange={val => updateForm("weight_class", val)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Catégorie de poids" />
                </SelectTrigger>
                <SelectContent>
                  {weightClasses.map(w => (
                    <SelectItem key={w} value={w}>
                      {w}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {/* Gender */}
              <Select value={form.gender} onValueChange={val => updateForm("gender", val)}>
                <SelectTrigger>
                  <SelectValue placeholder="Genre" />
                </SelectTrigger>
                <SelectContent>
                  {genders.map(g => (
                    <SelectItem key={g} value={g}>
                      {g}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {/* Stance */}
              <Select value={form.stance} onValueChange={val => updateForm("stance", val)}>
                <SelectTrigger>
                  <SelectValue placeholder="Garde" />
                </SelectTrigger>
                <SelectContent>
                  {stances.map(s => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={submitFighter}
              disabled={submitting}
              className="w-full"
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" /> Ajout…
                </>
              ) : (
                "Ajouter"
              )}
            </Button>
          </CardFooter>
        </Card>

        {/* List */}
        <Card>
          <CardHeader>
            <CardTitle>Liste des combattants ({fighters.length})</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {loading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            ) : fighters.length === 0 ? (
              <p className="text-center text-gray-500">Aucun combattant enregistré.</p>
            ) : (
              fighters.map(f => (
                <motion.div
                  key={f.name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-between items-center p-2 rounded-lg bg-white/50 backdrop-blur-md"
                >
                  <div className="flex flex-col">
                    <span className="font-medium">{f.name}</span>
                    <span className="text-xs text-gray-600">
                      {f.weight_class} • {f.gender}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {new Date(f.created_at ?? "").toLocaleDateString()}
                    </Badge>
                    <Button
                      variant="destructive"
                      size="icon"
                      onClick={() => deleteFighter(f.name)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </motion.div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  )
}

import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { UploadCloud, Activity, Mic, ShieldAlert, CheckCircle2 } from 'lucide-react'
import axios from 'axios'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = e.target.files[0]
      setFile(selected)
      setPrediction(null)
      setError(null)
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      setAudioUrl(URL.createObjectURL(selected))
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const analyzeAudio = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      // Assuming our FastAPI backend is running on local port 8000
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      setPrediction(response.data.prediction)
      setConfidence(response.data.confidence)
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to connect to the prediction server.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 relative">
      
      {/* Decorative Background Elements */}
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-secondary/20 rounded-full blur-[120px] pointer-events-none" />

      <motion.div 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="text-center mb-12 z-10"
      >
        <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-4">
          Vocal <span className="neon-text">Metrics</span>
        </h1>
        <p className="text-gray-400 text-lg md:text-xl max-w-xl mx-auto font-light tracking-wide">
          Advanced Speech Emotion Recognition powered by Deep Neural Networks.
        </p>
      </motion.div>

      <motion.div 
        className="glass-panel w-full max-w-2xl p-8 relative z-10"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        {!prediction ? (
          <div className="flex flex-col items-center">
            
            <input 
              type="file" 
              accept="audio/*" 
              className="hidden" 
              ref={fileInputRef}
              onChange={handleFileChange}
            />

            <motion.div 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleUploadClick}
              className={`w-full border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                file ? 'border-primary bg-primary/5' : 'border-white/20 hover:border-primary/50 bg-white/5'
              }`}
            >
              {file ? (
                <>
                  <Mic className="w-16 h-16 text-primary mb-4 animate-pulse" />
                  <p className="text-xl font-medium text-white">{file.name}</p>
                  <p className="text-gray-400 mt-2">Ready for analysis</p>
                </>
              ) : (
                <>
                  <UploadCloud className="w-16 h-16 text-gray-400 mb-4" />
                  <p className="text-xl font-medium text-gray-300">Select Audio File</p>
                  <p className="text-sm text-gray-500 mt-2">WAV, MP3, or FLAC up to 10MB</p>
                </>
              )}
            </motion.div>

            {error && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-4 w-full bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-3 text-red-200"
              >
                <ShieldAlert />
                <span>{error}</span>
              </motion.div>
            )}

            {audioUrl && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-6 w-full">
                <audio src={audioUrl} controls className="w-full rounded-full" />
              </motion.div>
            )}

            <motion.button
              onClick={analyzeAudio}
              disabled={!file || isAnalyzing}
              whileHover={file && !isAnalyzing ? { scale: 1.05 } : {}}
              whileTap={file && !isAnalyzing ? { scale: 0.95 } : {}}
              className={`mt-8 px-8 py-4 rounded-full font-bold text-lg tracking-wider w-full md:w-auto transition-all ${
                !file || isAnalyzing 
                  ? 'bg-white/10 text-white/30 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-primary to-secondary text-white shadow-[0_0_20px_rgba(0,240,255,0.4)] hover:shadow-[0_0_30px_rgba(0,240,255,0.6)]'
              }`}
            >
              {isAnalyzing ? (
                <span className="flex items-center justify-center gap-2">
                  <Activity className="animate-spin" /> ANALYZING PATTERNS...
                </span>
              ) : (
                'INITIALIZE ANALYSIS'
              )}
            </motion.button>

          </div>
        ) : (
          <AnimatePresence>
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center py-8"
            >
              <CheckCircle2 className="w-20 h-20 text-success mb-6 shadow-glow" />
              <h2 className="text-gray-400 uppercase tracking-widest text-sm mb-2">Detected Emotion</h2>
              <div className="text-6xl font-bold capitalize neon-text mb-6">
                {prediction}
              </div>
              <div className="bg-white/5 border border-white/10 rounded-lg px-6 py-3 flex items-center gap-4">
                <span className="text-gray-400">Confidence Score:</span>
                <span className="text-2xl font-mono text-white">{confidence}</span>
              </div>
              
              {audioUrl && (
                <div className="mt-6 w-full max-w-sm">
                  <audio src={audioUrl} controls className="w-full rounded-full" />
                </div>
              )}

              <button 
                onClick={() => { 
                  setPrediction(null); 
                  setFile(null); 
                  if (audioUrl) URL.revokeObjectURL(audioUrl);
                  setAudioUrl(null);
                }}
                className="mt-10 text-gray-400 hover:text-white transition-colors underline underline-offset-4"
              >
                Analyze Another File
              </button>
            </motion.div>
          </AnimatePresence>
        )}
      </motion.div>
    </div>
  )
}

export default App

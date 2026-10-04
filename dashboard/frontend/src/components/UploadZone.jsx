import { useRef, useState } from 'react'

export default function UploadZone({ onFile, loading, disabled }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const [name, setName] = useState('')

  function pick(file) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Please choose a .csv file')
      return
    }
    setName(file.name)
    onFile(file)
  }

  return (
    <div
      className={`fx-upload ${drag ? 'drag' : ''} ${disabled ? 'disabled' : ''}`}
      onClick={() => !disabled && !loading && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        if (!disabled) pick(e.dataTransfer.files?.[0])
      }}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        hidden
        onChange={(e) => {
          pick(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      <div className="k">UPLOAD TRANSACTIONS</div>
      <div className="fx-upload-main">
        {loading ? 'Scoring...' : name ? name : 'Drop a CSV here or click to browse'}
      </div>
      <div className="hint">Columns: Time, V1..V28, Amount. No label column.</div>
    </div>
  )
}

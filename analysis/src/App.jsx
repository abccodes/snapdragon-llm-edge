import React, { useState, useMemo, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, Legend, LineChart, Line } from 'recharts';

// Color palette for models
const colorPalette = [
  "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", 
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1"
];

// Parse CSV string to array of objects
const parseCSV = (csvString, fileName) => {
  const lines = csvString.trim().split('\n');
  if (lines.length < 2) return [];
  
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  
  return lines.slice(1).map((line, idx) => {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let char of line) {
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        values.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    values.push(current.trim());
    
    const obj = { _source: fileName };
    headers.forEach((header, i) => {
      let val = values[i] || '';
      val = val.replace(/"/g, '');
      const num = parseFloat(val);
      obj[header] = isNaN(num) ? val : num;
    });
    return obj;
  });
};

// Process raw data to add normalized scores
const processData = (data) => {
  if (data.length === 0) return [];
  
  // Filter valid runs
  const valid = data.filter(d => 
    d.accuracy > 0 && 
    (d.avg_decode_speed > 0 || d.decode_speed > 0)
  );
  
  if (valid.length === 0) return [];
  
  // Normalize speed field name
  valid.forEach(d => {
    if (!d.decode_speed && d.avg_decode_speed) {
      d.decode_speed = d.avg_decode_speed;
    }
  });
  
  // Check if BLEURT scores are available
  const hasBleurt = valid.some(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score));
  
  // Calculate normalized scores
  const minAcc = Math.min(...valid.map(d => d.accuracy));
  const maxAcc = Math.max(...valid.map(d => d.accuracy));
  const minSpeed = Math.min(...valid.map(d => d.decode_speed));
  const maxSpeed = Math.max(...valid.map(d => d.decode_speed));
  
  // BLEURT scores are typically negative (higher = better)
  let minBleurt = 0, maxBleurt = 0, bleurtRange = 1;
  if (hasBleurt) {
    const bleurtValues = valid.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score)).map(d => d.bleurt_score);
    minBleurt = Math.min(...bleurtValues);
    maxBleurt = Math.max(...bleurtValues);
    bleurtRange = maxBleurt - minBleurt || 1;
  }
  
  const accRange = maxAcc - minAcc || 1;
  const speedRange = maxSpeed - minSpeed || 1;
  
  return valid.map(d => {
    const norm_accuracy = (d.accuracy - minAcc) / accRange;
    const norm_speed = (d.decode_speed - minSpeed) / speedRange;
    const norm_bleurt = hasBleurt && d.bleurt_score !== undefined && !isNaN(d.bleurt_score)
      ? (d.bleurt_score - minBleurt) / bleurtRange 
      : null;
    
    // Combined score: if BLEURT available, use 3-way average
    let combined;
    if (norm_bleurt !== null) {
      combined = (norm_accuracy + norm_speed + norm_bleurt) / 3;
    } else {
      combined = (norm_accuracy + norm_speed) / 2;
    }
    
    return {
      ...d,
      norm_accuracy,
      norm_speed,
      norm_bleurt,
      combined,
      hasBleurt: norm_bleurt !== null,
      model_short: d.model ? d.model.replace(/-Q4_K_M\.gguf|\.gguf/g, '').substring(0, 20) : 'unknown'
    };
  });
};

const CustomTooltip = ({ active, payload, modelColors }) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{
        background: 'rgba(15, 23, 42, 0.95)',
        border: '1px solid rgba(148, 163, 184, 0.3)',
        borderRadius: '8px',
        padding: '12px 16px',
        fontSize: '13px',
        color: '#e2e8f0',
        boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        maxWidth: '320px'
      }}>
        <div style={{ fontWeight: 700, marginBottom: 8, color: modelColors[d.model_short] || '#fff' }}>
          Run #{d.run_id} • {d.model_short}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
          <span style={{ color: '#94a3b8' }}>Accuracy:</span><span style={{ fontWeight: 600 }}>{(d.accuracy * 100).toFixed(1)}%</span>
          <span style={{ color: '#94a3b8' }}>Speed:</span><span style={{ fontWeight: 600 }}>{d.decode_speed?.toFixed(1)} tok/s</span>
          {d.bleurt_score !== undefined && !isNaN(d.bleurt_score) && (
            <><span style={{ color: '#94a3b8' }}>BLEURT:</span><span style={{ fontWeight: 600, color: '#06b6d4' }}>{d.bleurt_score?.toFixed(3)}</span></>
          )}
          <span style={{ color: '#94a3b8' }}>Combined:</span><span style={{ fontWeight: 600, color: '#fbbf24' }}>{d.combined?.toFixed(3)}</span>
          {d.temperature !== undefined && <><span style={{ color: '#94a3b8' }}>Temp:</span><span>{d.temperature}</span></>}
          {d.threads !== undefined && <><span style={{ color: '#94a3b8' }}>Threads:</span><span>{d.threads}</span></>}
          {d.repeat_penalty !== undefined && <><span style={{ color: '#94a3b8' }}>Rep Pen:</span><span>{d.repeat_penalty}</span></>}
          {d._source && <><span style={{ color: '#94a3b8' }}>Source:</span><span style={{ fontSize: '11px' }}>{d._source}</span></>}
        </div>
      </div>
    );
  }
  return null;
};

function App() {
  const [files, setFiles] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedSource, setSelectedSource] = useState('all');

  const handleFileUpload = useCallback((e) => {
    const uploadedFiles = Array.from(e.target.files);
    uploadedFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const csvString = event.target.result;
        const parsed = parseCSV(csvString, file.name);
        setFiles(prev => [...prev, file.name]);
        setRawData(prev => [...prev, ...parsed]);
      };
      reader.readAsText(file);
    });
  }, []);

  const removeFile = useCallback((fileName) => {
    setFiles(prev => prev.filter(f => f !== fileName));
    setRawData(prev => prev.filter(d => d._source !== fileName));
  }, []);

  const processedData = useMemo(() => processData(rawData), [rawData]);
  const hasBleurtData = useMemo(() => processedData.some(d => d.hasBleurt), [processedData]);
  const models = useMemo(() => [...new Set(processedData.map(d => d.model_short))].sort(), [processedData]);
  
  const modelColors = useMemo(() => {
    const colors = {};
    models.forEach((m, i) => { colors[m] = colorPalette[i % colorPalette.length]; });
    return colors;
  }, [models]);

  const filteredData = useMemo(() => {
    let data = processedData;
    if (selectedModel !== 'all') data = data.filter(d => d.model_short === selectedModel);
    if (selectedSource !== 'all') data = data.filter(d => d._source === selectedSource);
    return data;
  }, [processedData, selectedModel, selectedSource]);

  const threadStats = useMemo(() => {
    if (filteredData.length === 0 || filteredData[0].threads === undefined) return [];
    const groups = {};
    filteredData.forEach(d => {
      if (d.threads === undefined) return;
      if (!groups[d.threads]) groups[d.threads] = { threads: d.threads, accuracy: [], speed: [], bleurt: [], combined: [] };
      groups[d.threads].accuracy.push(d.accuracy);
      groups[d.threads].speed.push(d.decode_speed);
      if (d.bleurt_score !== undefined && !isNaN(d.bleurt_score)) groups[d.threads].bleurt.push(d.bleurt_score);
      groups[d.threads].combined.push(d.combined);
    });
    return Object.values(groups).map(g => ({
      threads: g.threads,
      accuracy: g.accuracy.reduce((a,b) => a+b, 0) / g.accuracy.length,
      speed: g.speed.reduce((a,b) => a+b, 0) / g.speed.length,
      bleurt: g.bleurt.length > 0 ? g.bleurt.reduce((a,b) => a+b, 0) / g.bleurt.length : null,
      combined: g.combined.reduce((a,b) => a+b, 0) / g.combined.length,
    })).sort((a,b) => a.threads - b.threads);
  }, [filteredData]);

  const tempStats = useMemo(() => {
    if (filteredData.length === 0 || filteredData[0].temperature === undefined) return [];
    const groups = {};
    filteredData.forEach(d => {
      if (d.temperature === undefined) return;
      if (!groups[d.temperature]) groups[d.temperature] = { temp: d.temperature, accuracy: [], speed: [], bleurt: [], combined: [] };
      groups[d.temperature].accuracy.push(d.accuracy);
      groups[d.temperature].speed.push(d.decode_speed);
      if (d.bleurt_score !== undefined && !isNaN(d.bleurt_score)) groups[d.temperature].bleurt.push(d.bleurt_score);
      groups[d.temperature].combined.push(d.combined);
    });
    return Object.values(groups).map(g => ({
      temp: g.temp,
      accuracy: g.accuracy.reduce((a,b) => a+b, 0) / g.accuracy.length,
      speed: g.speed.reduce((a,b) => a+b, 0) / g.speed.length,
      bleurt: g.bleurt.length > 0 ? g.bleurt.reduce((a,b) => a+b, 0) / g.bleurt.length : null,
      combined: g.combined.reduce((a,b) => a+b, 0) / g.combined.length,
    })).sort((a,b) => a.temp - b.temp);
  }, [filteredData]);

  const modelStats = useMemo(() => {
    if (processedData.length === 0) return [];
    const groups = {};
    const dataToUse = selectedSource === 'all' ? processedData : processedData.filter(d => d._source === selectedSource);
    dataToUse.forEach(d => {
      if (!groups[d.model_short]) groups[d.model_short] = { model: d.model_short, accuracy: [], speed: [], bleurt: [], combined: [] };
      groups[d.model_short].accuracy.push(d.accuracy);
      groups[d.model_short].speed.push(d.decode_speed);
      if (d.bleurt_score !== undefined && !isNaN(d.bleurt_score)) groups[d.model_short].bleurt.push(d.bleurt_score);
      groups[d.model_short].combined.push(d.combined);
    });
    return Object.values(groups).map(g => ({
      model: g.model,
      accuracy: g.accuracy.reduce((a,b) => a+b, 0) / g.accuracy.length,
      speed: g.speed.reduce((a,b) => a+b, 0) / g.speed.length,
      bleurt: g.bleurt.length > 0 ? g.bleurt.reduce((a,b) => a+b, 0) / g.bleurt.length : null,
      combined: g.combined.reduce((a,b) => a+b, 0) / g.combined.length,
      maxCombined: Math.max(...g.combined),
    })).sort((a,b) => b.combined - a.combined);
  }, [processedData, selectedSource]);

  const topRuns = useMemo(() => [...filteredData].sort((a,b) => b.combined - a.combined).slice(0, 5), [filteredData]);

  const bleurtStats = useMemo(() => {
    if (!hasBleurtData || filteredData.length === 0) return null;
    const withBleurt = filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score));
    if (withBleurt.length === 0) return null;
    const scores = withBleurt.map(d => d.bleurt_score);
    return { min: Math.min(...scores), max: Math.max(...scores), avg: scores.reduce((a,b) => a+b, 0) / scores.length, count: scores.length };
  }, [filteredData, hasBleurtData]);

  // Empty state
  if (rawData.length === 0) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
        color: '#e2e8f0',
        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '32px',
      }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: 800,
          background: 'linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: 16,
        }}>
          LLM Hyperparameter Analyzer
        </h1>
        <p style={{ color: '#64748b', marginBottom: 32 }}>Upload CSV files from your hyperparameter search</p>
        
        <label style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '48px 64px',
          border: '2px dashed #475569',
          borderRadius: '16px',
          cursor: 'pointer',
          background: 'rgba(30, 41, 59, 0.5)',
        }}
        onMouseOver={(e) => e.currentTarget.style.borderColor = '#10b981'}
        onMouseOut={(e) => e.currentTarget.style.borderColor = '#475569'}
        >
          <svg width="48" height="48" fill="none" stroke="#64748b" viewBox="0 0 24 24" style={{ marginBottom: 16 }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <span style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8 }}>Drop CSV files here</span>
          <span style={{ color: '#64748b', fontSize: '0.9rem' }}>or click to browse</span>
          <input type="file" accept=".csv" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
        
        <p style={{ color: '#475569', marginTop: 24, fontSize: '0.85rem', textAlign: 'center' }}>
          Required: run_id, model, accuracy, avg_decode_speed<br/>
          Optional: bleurt_score, temperature, threads, repeat_penalty
        </p>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
      color: '#e2e8f0',
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
      padding: '32px',
    }}>
      {/* Header */}
      <div style={{ marginBottom: 32, textAlign: 'center' }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: 800,
          background: 'linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: 8,
        }}>
          LLM Hyperparameter Analysis
        </h1>
        <p style={{ color: '#64748b' }}>
          {filteredData.length} valid runs • {files.length} file(s)
          {hasBleurtData && ' • BLEURT scores included'}
        </p>
      </div>

      {/* File Management */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 24, alignItems: 'center' }}>
        <span style={{ color: '#94a3b8', fontSize: '0.85rem', marginRight: 8 }}>Files:</span>
        {files.map(f => (
          <span key={f} style={{
            background: 'rgba(59, 130, 246, 0.2)',
            border: '1px solid rgba(59, 130, 246, 0.4)',
            borderRadius: 6,
            padding: '4px 10px',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}>
            {f}
            <button onClick={() => removeFile(f)} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: 0, fontSize: '1rem' }}>×</button>
          </span>
        ))}
        <label style={{ background: 'rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: 6, padding: '4px 12px', fontSize: '0.85rem', cursor: 'pointer' }}>
          + Add CSV
          <input type="file" accept=".csv" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
      </div>

      {/* Top Runs */}
      <div style={{ marginBottom: 40 }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: 16, color: '#fbbf24' }}>
          🏆 Top 5 Configurations {hasBleurtData ? '(Speed + Accuracy + BLEURT)' : '(Speed + Accuracy)'}
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          {topRuns.map((run, i) => (
            <div key={`${run._source}-${run.run_id}`} style={{
              background: i === 0 ? 'linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.05))' : 'rgba(30, 41, 59, 0.6)',
              border: i === 0 ? '2px solid rgba(251, 191, 36, 0.4)' : '1px solid rgba(148, 163, 184, 0.15)',
              borderRadius: 12,
              padding: 20,
              position: 'relative',
            }}>
              <div style={{ position: 'absolute', top: 12, right: 12, background: i === 0 ? '#fbbf24' : '#475569', color: i === 0 ? '#0f172a' : '#e2e8f0', borderRadius: 20, padding: '4px 12px', fontSize: '0.75rem', fontWeight: 700 }}>
                #{i + 1}
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: modelColors[run.model_short], marginBottom: 8 }}>{run.model_short}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: '0.85rem' }}>
                <div><span style={{ color: '#64748b' }}>Score: </span><span style={{ fontWeight: 700, color: '#fbbf24' }}>{run.combined?.toFixed(3)}</span></div>
                <div><span style={{ color: '#64748b' }}>Acc: </span><span style={{ fontWeight: 600 }}>{(run.accuracy * 100).toFixed(0)}%</span></div>
                <div><span style={{ color: '#64748b' }}>Speed: </span><span style={{ fontWeight: 600 }}>{run.decode_speed?.toFixed(1)}</span></div>
                {run.bleurt_score !== undefined && !isNaN(run.bleurt_score) && (
                  <div><span style={{ color: '#64748b' }}>BLEURT: </span><span style={{ fontWeight: 600, color: '#06b6d4' }}>{run.bleurt_score?.toFixed(3)}</span></div>
                )}
                {run.temperature !== undefined && <div><span style={{ color: '#64748b' }}>Temp: </span><span>{run.temperature}</span></div>}
                {run.threads !== undefined && <div><span style={{ color: '#64748b' }}>Threads: </span><span>{run.threads}</span></div>}
                {run.repeat_penalty !== undefined && <div><span style={{ color: '#64748b' }}>Rep: </span><span>{run.repeat_penalty}</span></div>}
              </div>
              {files.length > 1 && <div style={{ marginTop: 8, fontSize: '0.75rem', color: '#64748b' }}>{run._source}</div>}
            </div>
          ))}
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 32, flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <label style={{ color: '#94a3b8', fontSize: '0.85rem', marginRight: 8 }}>Model:</label>
          <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 6, padding: '8px 12px', color: '#e2e8f0', cursor: 'pointer' }}>
            <option value="all">All Models</option>
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        {files.length > 1 && (
          <div>
            <label style={{ color: '#94a3b8', fontSize: '0.85rem', marginRight: 8 }}>File:</label>
            <select value={selectedSource} onChange={e => setSelectedSource(e.target.value)} style={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 6, padding: '8px 12px', color: '#e2e8f0', cursor: 'pointer' }}>
              <option value="all">All Files</option>
              {files.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: 24, marginBottom: 32 }}>
        
        {/* Speed vs Accuracy */}
        <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>Speed vs Accuracy</h3>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="decode_speed" name="Speed" unit=" tok/s" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Decode Speed (tok/s)', position: 'bottom', fill: '#94a3b8', fontSize: 12 }} />
              <YAxis type="number" dataKey="accuracy" name="Accuracy" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
              <Scatter data={filteredData} fill="#3b82f6">
                {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#3b82f6'} fillOpacity={0.8} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* BLEURT vs Accuracy */}
        {hasBleurtData && (
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>BLEURT vs Accuracy</h3>
            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="bleurt_score" name="BLEURT" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'BLEURT Score', position: 'bottom', fill: '#94a3b8', fontSize: 12 }} />
                <YAxis type="number" dataKey="accuracy" name="Accuracy" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                <Scatter data={filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score))} fill="#06b6d4">
                  {filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score)).map((entry, index) => <Cell key={`cell-bleurt-${index}`} fill={modelColors[entry.model_short] || '#06b6d4'} fillOpacity={0.8} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Model Comparison */}
        <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>Model Combined Score</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 1]} stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis type="category" dataKey="model" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} width={95} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} labelStyle={{ color: '#e2e8f0' }} />
              <Bar dataKey="combined" name="Combined Score" radius={[0, 4, 4, 0]}>
                {modelStats.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#3b82f6'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model BLEURT */}
        {hasBleurtData && (
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>Model BLEURT Scores</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={modelStats.filter(m => m.bleurt !== null)} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis type="category" dataKey="model" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} width={95} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} labelStyle={{ color: '#e2e8f0' }} formatter={(value) => value?.toFixed(3)} />
                <Bar dataKey="bleurt" name="Avg BLEURT" fill="#06b6d4" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Thread Impact */}
        {threadStats.length > 1 && (
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>Thread Count Impact</h3>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={threadStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="threads" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Threads', position: 'bottom', fill: '#94a3b8', fontSize: 12 }} />
                <YAxis yAxisId="left" stroke="#10b981" tick={{ fill: '#10b981', fontSize: 12 }} />
                <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" tick={{ fill: '#f59e0b', fontSize: 12 }} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} labelStyle={{ color: '#e2e8f0' }} />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="speed" name="Speed (tok/s)" stroke="#10b981" strokeWidth={3} dot={{ r: 6, fill: '#10b981' }} />
                <Line yAxisId="right" type="monotone" dataKey="combined" name="Combined" stroke="#f59e0b" strokeWidth={3} dot={{ r: 6, fill: '#f59e0b' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Temperature Impact */}
        {tempStats.length > 1 && (
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>Temperature Impact</h3>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={tempStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="temp" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Temperature', position: 'bottom', fill: '#94a3b8', fontSize: 12 }} />
                <YAxis yAxisId="left" stroke="#8b5cf6" tick={{ fill: '#8b5cf6', fontSize: 12 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" tick={{ fill: '#f59e0b', fontSize: 12 }} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} labelStyle={{ color: '#e2e8f0' }} />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 6, fill: '#8b5cf6' }} />
                <Line yAxisId="right" type="monotone" dataKey="combined" name="Combined" stroke="#f59e0b" strokeWidth={3} dot={{ r: 6, fill: '#f59e0b' }} />
                {hasBleurtData && tempStats.some(t => t.bleurt !== null) && (
                  <Line yAxisId="right" type="monotone" dataKey="bleurt" name="BLEURT" stroke="#06b6d4" strokeWidth={2} dot={{ r: 4, fill: '#06b6d4' }} strokeDasharray="5 5" />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      <div style={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1))', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: 16, padding: 24 }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: 16, color: '#10b981' }}>📊 Summary</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16 }}>
          <div>
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Runs</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{filteredData.length}</div>
          </div>
          <div>
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Best Accuracy</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#8b5cf6' }}>{(Math.max(...filteredData.map(d => d.accuracy)) * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Best Speed</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>{Math.max(...filteredData.map(d => d.decode_speed)).toFixed(1)} tok/s</div>
          </div>
          {bleurtStats && (
            <div>
              <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Best BLEURT</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#06b6d4' }}>{bleurtStats.max.toFixed(3)}</div>
            </div>
          )}
          <div>
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Best Combined</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#fbbf24' }}>{Math.max(...filteredData.map(d => d.combined)).toFixed(3)}</div>
          </div>
        </div>
        {hasBleurtData && (
          <div style={{ marginTop: 16, padding: '12px 16px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: 8, border: '1px solid rgba(6, 182, 212, 0.3)' }}>
            <span style={{ color: '#06b6d4', fontWeight: 600 }}>ℹ️ Combined Score:</span>
            <span style={{ color: '#94a3b8', marginLeft: 8 }}>(accuracy + speed + BLEURT) / 3 — all normalized 0-1</span>
          </div>
        )}
      </div>

      <div style={{ textAlign: 'center', marginTop: 40, color: '#475569', fontSize: '0.85rem' }}>
        LLM Hyperparameter Analyzer • {filteredData.length} runs {hasBleurtData && `• ${bleurtStats?.count || 0} with BLEURT`}
      </div>
    </div>
  );
}

export default App;

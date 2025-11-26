import React, { useState, useMemo, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, Legend, LineChart, Line } from 'recharts';

const colorPalette = [
  "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", 
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
  "#14b8a6", "#a855f7", "#eab308", "#22c55e", "#0ea5e9"
];

const parseCSV = (csvString, fileName) => {
  const lines = csvString.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  
  return lines.slice(1).map((line) => {
    const values = [];
    let current = '';
    let inQuotes = false;
    for (let char of line) {
      if (char === '"') inQuotes = !inQuotes;
      else if (char === ',' && !inQuotes) { values.push(current.trim()); current = ''; }
      else current += char;
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

const processData = (data) => {
  if (data.length === 0) return [];
  
  const valid = data.filter(d => d.accuracy > 0 && (d.avg_decode_speed > 0 || d.decode_speed > 0));
  if (valid.length === 0) return [];
  
  valid.forEach(d => {
    if (!d.decode_speed && d.avg_decode_speed) d.decode_speed = d.avg_decode_speed;
    if (!d.prefill_speed && d.avg_prefill_speed) d.prefill_speed = d.avg_prefill_speed;
    if (!d.total_speed && d.avg_total_speed) d.total_speed = d.avg_total_speed;
  });
  
  const hasBleurt = valid.some(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score));
  
  const minAcc = Math.min(...valid.map(d => d.accuracy));
  const maxAcc = Math.max(...valid.map(d => d.accuracy));
  const minSpeed = Math.min(...valid.map(d => d.decode_speed));
  const maxSpeed = Math.max(...valid.map(d => d.decode_speed));
  
  let minBleurt = 0, maxBleurt = 0;
  if (hasBleurt) {
    const bleurtValues = valid.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score)).map(d => d.bleurt_score);
    minBleurt = Math.min(...bleurtValues);
    maxBleurt = Math.max(...bleurtValues);
  }
  
  const accRange = maxAcc - minAcc || 1;
  const speedRange = maxSpeed - minSpeed || 1;
  const bleurtRange = maxBleurt - minBleurt || 1;
  
  return valid.map(d => {
    const norm_accuracy = (d.accuracy - minAcc) / accRange;
    const norm_speed = (d.decode_speed - minSpeed) / speedRange;
    const norm_bleurt = hasBleurt && d.bleurt_score !== undefined && !isNaN(d.bleurt_score)
      ? (d.bleurt_score - minBleurt) / bleurtRange : null;
    
    const combined = norm_bleurt !== null
      ? (norm_accuracy + norm_speed + norm_bleurt) / 3
      : (norm_accuracy + norm_speed) / 2;
    
    return {
      ...d,
      norm_accuracy, norm_speed, norm_bleurt, combined,
      hasBleurt: norm_bleurt !== null,
      model_short: d.model ? d.model.replace(/-Q4_K_M\.gguf|\.gguf/g, '').substring(0, 25) : 'unknown'
    };
  });
};

const aggregateByParam = (data, param) => {
  if (data.length === 0 || data[0][param] === undefined) return [];
  const groups = {};
  data.forEach(d => {
    const key = d[param];
    if (key === undefined || key === '') return;
    if (!groups[key]) groups[key] = { [param]: key, accuracy: [], speed: [], bleurt: [], combined: [], prefill: [], runtime: [] };
    groups[key].accuracy.push(d.accuracy);
    groups[key].speed.push(d.decode_speed);
    groups[key].combined.push(d.combined);
    if (d.bleurt_score !== undefined && !isNaN(d.bleurt_score)) groups[key].bleurt.push(d.bleurt_score);
    if (d.prefill_speed) groups[key].prefill.push(d.prefill_speed);
    if (d.runtime_seconds) groups[key].runtime.push(d.runtime_seconds);
  });
  return Object.values(groups).map(g => ({
    [param]: g[param],
    accuracy: g.accuracy.reduce((a,b) => a+b, 0) / g.accuracy.length,
    speed: g.speed.reduce((a,b) => a+b, 0) / g.speed.length,
    bleurt: g.bleurt.length > 0 ? g.bleurt.reduce((a,b) => a+b, 0) / g.bleurt.length : null,
    combined: g.combined.reduce((a,b) => a+b, 0) / g.combined.length,
    prefill: g.prefill.length > 0 ? g.prefill.reduce((a,b) => a+b, 0) / g.prefill.length : null,
    runtime: g.runtime.length > 0 ? g.runtime.reduce((a,b) => a+b, 0) / g.runtime.length : null,
    count: g.accuracy.length
  })).sort((a,b) => {
    const aVal = a[param], bVal = b[param];
    if (typeof aVal === 'number') return aVal - bVal;
    return String(aVal).localeCompare(String(bVal));
  });
};

const CustomTooltip = ({ active, payload, modelColors }) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{ background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(148, 163, 184, 0.3)', borderRadius: '8px', padding: '12px 16px', fontSize: '12px', color: '#e2e8f0', boxShadow: '0 4px 20px rgba(0,0,0,0.4)', maxWidth: '380px' }}>
        <div style={{ fontWeight: 700, marginBottom: 8, color: modelColors?.[d.model_short] || '#fff', fontSize: '13px' }}>
          Run #{d.run_id} • {d.model_short}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px 12px' }}>
          <span style={{ color: '#94a3b8' }}>Accuracy:</span><span style={{ fontWeight: 600 }}>{(d.accuracy * 100).toFixed(1)}%</span><span></span>
          <span style={{ color: '#94a3b8' }}>Decode:</span><span style={{ fontWeight: 600 }}>{d.decode_speed?.toFixed(1)} tok/s</span><span></span>
          {d.prefill_speed && <><span style={{ color: '#94a3b8' }}>Prefill:</span><span>{d.prefill_speed?.toFixed(1)} tok/s</span><span></span></>}
          {d.bleurt_score !== undefined && !isNaN(d.bleurt_score) && <><span style={{ color: '#94a3b8' }}>BLEURT:</span><span style={{ color: '#06b6d4' }}>{d.bleurt_score?.toFixed(3)}</span><span></span></>}
          <span style={{ color: '#94a3b8' }}>Combined:</span><span style={{ fontWeight: 600, color: '#fbbf24' }}>{d.combined?.toFixed(3)}</span><span></span>
          {d.temperature !== undefined && <><span style={{ color: '#94a3b8' }}>Temp:</span><span>{d.temperature}</span><span></span></>}
          {d.threads !== undefined && <><span style={{ color: '#94a3b8' }}>Threads:</span><span>{d.threads}</span><span></span></>}
          {d.repeat_penalty !== undefined && <><span style={{ color: '#94a3b8' }}>Rep Pen:</span><span>{d.repeat_penalty}</span><span></span></>}
          {d.top_p !== undefined && <><span style={{ color: '#94a3b8' }}>Top-P:</span><span>{d.top_p}</span><span></span></>}
          {d.top_k !== undefined && <><span style={{ color: '#94a3b8' }}>Top-K:</span><span>{d.top_k}</span><span></span></>}
          {d.ctx_size !== undefined && <><span style={{ color: '#94a3b8' }}>Ctx:</span><span>{d.ctx_size}</span><span></span></>}
          {d.batch_size !== undefined && <><span style={{ color: '#94a3b8' }}>Batch:</span><span>{d.batch_size}</span><span></span></>}
          {d.ubatch_size !== undefined && <><span style={{ color: '#94a3b8' }}>UBatch:</span><span>{d.ubatch_size}</span><span></span></>}
          {d.flash_attn !== undefined && <><span style={{ color: '#94a3b8' }}>Flash:</span><span>{d.flash_attn}</span><span></span></>}
          {d.ctk !== undefined && <><span style={{ color: '#94a3b8' }}>KV-K:</span><span>{d.ctk}</span><span></span></>}
          {d.ctv !== undefined && <><span style={{ color: '#94a3b8' }}>KV-V:</span><span>{d.ctv}</span><span></span></>}
          {d.use_mmap !== undefined && <><span style={{ color: '#94a3b8' }}>MMap:</span><span>{d.use_mmap}</span><span></span></>}
          {d.poll_level !== undefined && <><span style={{ color: '#94a3b8' }}>Poll:</span><span>{d.poll_level}</span><span></span></>}
          {d.runtime_seconds !== undefined && <><span style={{ color: '#94a3b8' }}>Runtime:</span><span>{d.runtime_seconds}s</span><span></span></>}
        </div>
      </div>
    );
  }
  return null;
};

const ParamChart = ({ data, param, title, modelColors }) => {
  if (!data || data.length < 2) return null;
  const hasBleurt = data.some(d => d.bleurt !== null);
  
  return (
    <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey={param} stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis yAxisId="left" stroke="#10b981" tick={{ fill: '#10b981', fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" tick={{ fill: '#f59e0b', fontSize: 11 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8, fontSize: 12 }} />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="speed" name="Speed (tok/s)" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
          <Line yAxisId="right" type="monotone" dataKey="combined" name="Combined" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} />
          <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
          {hasBleurt && <Line yAxisId="right" type="monotone" dataKey="bleurt" name="BLEURT (norm)" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="5 5" />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const BarParamChart = ({ data, param, title, metric = 'combined' }) => {
  if (!data || data.length < 1) return null;
  
  return (
    <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 24 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#94a3b8' }}>{title}</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey={param} stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10, angle: -45, textAnchor: 'end' }} height={60} />
          <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} />
          <Bar dataKey={metric} name={metric === 'combined' ? 'Combined Score' : metric} fill="#3b82f6" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

function App() {
  const [files, setFiles] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedSource, setSelectedSource] = useState('all');

  const handleFileUpload = useCallback((e) => {
    Array.from(e.target.files).forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const parsed = parseCSV(event.target.result, file.name);
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

  // Aggregate stats for all parameters
  const threadStats = useMemo(() => aggregateByParam(filteredData, 'threads'), [filteredData]);
  const tempStats = useMemo(() => aggregateByParam(filteredData, 'temperature'), [filteredData]);
  const repPenStats = useMemo(() => aggregateByParam(filteredData, 'repeat_penalty'), [filteredData]);
  const topPStats = useMemo(() => aggregateByParam(filteredData, 'top_p'), [filteredData]);
  const topKStats = useMemo(() => aggregateByParam(filteredData, 'top_k'), [filteredData]);
  const ctxStats = useMemo(() => aggregateByParam(filteredData, 'ctx_size'), [filteredData]);
  const batchStats = useMemo(() => aggregateByParam(filteredData, 'batch_size'), [filteredData]);
  const ubatchStats = useMemo(() => aggregateByParam(filteredData, 'ubatch_size'), [filteredData]);
  const flashStats = useMemo(() => aggregateByParam(filteredData, 'flash_attn'), [filteredData]);
  const ctkStats = useMemo(() => aggregateByParam(filteredData, 'ctk'), [filteredData]);
  const ctvStats = useMemo(() => aggregateByParam(filteredData, 'ctv'), [filteredData]);
  const mmapStats = useMemo(() => aggregateByParam(filteredData, 'use_mmap'), [filteredData]);
  const pollStats = useMemo(() => aggregateByParam(filteredData, 'poll_level'), [filteredData]);

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
      count: g.accuracy.length
    })).sort((a,b) => b.combined - a.combined);
  }, [processedData, selectedSource]);

  const topRuns = useMemo(() => [...filteredData].sort((a,b) => b.combined - a.combined).slice(0, 6), [filteredData]);

  const stats = useMemo(() => {
    if (filteredData.length === 0) return null;
    const withBleurt = filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score));
    return {
      count: filteredData.length,
      maxAcc: Math.max(...filteredData.map(d => d.accuracy)),
      maxSpeed: Math.max(...filteredData.map(d => d.decode_speed)),
      maxCombined: Math.max(...filteredData.map(d => d.combined)),
      maxBleurt: withBleurt.length > 0 ? Math.max(...withBleurt.map(d => d.bleurt_score)) : null,
      avgBleurt: withBleurt.length > 0 ? withBleurt.reduce((a,b) => a + b.bleurt_score, 0) / withBleurt.length : null,
      avgRuntime: filteredData[0].runtime_seconds ? filteredData.reduce((a,b) => a + (b.runtime_seconds || 0), 0) / filteredData.length : null,
      bleurtCount: withBleurt.length
    };
  }, [filteredData]);

  // Empty state
  if (rawData.length === 0) {
    return (
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)', color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace", display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '32px' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 800, background: 'linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 16 }}>
          LLM Hyperparameter Analyzer
        </h1>
        <p style={{ color: '#64748b', marginBottom: 32 }}>Upload CSV files from your hyperparameter search</p>
        
        <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '48px 64px', border: '2px dashed #475569', borderRadius: '16px', cursor: 'pointer', background: 'rgba(30, 41, 59, 0.5)' }}
          onMouseOver={(e) => e.currentTarget.style.borderColor = '#10b981'}
          onMouseOut={(e) => e.currentTarget.style.borderColor = '#475569'}>
          <svg width="48" height="48" fill="none" stroke="#64748b" viewBox="0 0 24 24" style={{ marginBottom: 16 }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <span style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8 }}>Drop CSV files here</span>
          <span style={{ color: '#64748b', fontSize: '0.9rem' }}>or click to browse</span>
          <input type="file" accept=".csv" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
        
        <div style={{ color: '#475569', marginTop: 24, fontSize: '0.8rem', textAlign: 'center', maxWidth: 600 }}>
          <p style={{ marginBottom: 8 }}>Expected columns:</p>
          <p>run_id, model, accuracy, avg_decode_speed, temperature, threads, repeat_penalty, top_p, top_k, ctx_size, batch_size, ubatch_size, flash_attn, ctk, ctv, bleurt_score, runtime_seconds, ...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)', color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace", padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24, textAlign: 'center' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, background: 'linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 8 }}>
          LLM Hyperparameter Analysis
        </h1>
        <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
          {filteredData.length} runs • {files.length} file(s) {hasBleurtData && '• BLEURT included'}
        </p>
      </div>

      {/* File Management */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20, alignItems: 'center' }}>
        <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Files:</span>
        {files.map(f => (
          <span key={f} style={{ background: 'rgba(59, 130, 246, 0.2)', border: '1px solid rgba(59, 130, 246, 0.4)', borderRadius: 6, padding: '3px 8px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: 6 }}>
            {f}
            <button onClick={() => removeFile(f)} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: 0 }}>×</button>
          </span>
        ))}
        <label style={{ background: 'rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: 6, padding: '3px 10px', fontSize: '0.8rem', cursor: 'pointer' }}>
          + Add <input type="file" accept=".csv" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <label style={{ color: '#94a3b8', fontSize: '0.8rem', marginRight: 6 }}>Model:</label>
          <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 6, padding: '6px 10px', color: '#e2e8f0', cursor: 'pointer', fontSize: '0.85rem' }}>
            <option value="all">All Models</option>
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        {files.length > 1 && (
          <div>
            <label style={{ color: '#94a3b8', fontSize: '0.8rem', marginRight: 6 }}>File:</label>
            <select value={selectedSource} onChange={e => setSelectedSource(e.target.value)} style={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 6, padding: '6px 10px', color: '#e2e8f0', cursor: 'pointer', fontSize: '0.85rem' }}>
              <option value="all">All Files</option>
              {files.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* Top Runs */}
      <div style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 12, color: '#fbbf24' }}>
          🏆 Top Configurations {hasBleurtData ? '(Speed + Accuracy + BLEURT)' : '(Speed + Accuracy)'}
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12 }}>
          {topRuns.map((run, i) => (
            <div key={`${run._source}-${run.run_id}`} style={{
              background: i === 0 ? 'linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.05))' : 'rgba(30, 41, 59, 0.6)',
              border: i === 0 ? '2px solid rgba(251, 191, 36, 0.4)' : '1px solid rgba(148, 163, 184, 0.15)',
              borderRadius: 10, padding: 16, position: 'relative',
            }}>
              <div style={{ position: 'absolute', top: 10, right: 10, background: i === 0 ? '#fbbf24' : '#475569', color: i === 0 ? '#0f172a' : '#e2e8f0', borderRadius: 12, padding: '2px 10px', fontSize: '0.7rem', fontWeight: 700 }}>#{i + 1}</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: modelColors[run.model_short], marginBottom: 8 }}>{run.model_short}</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px 8px', fontSize: '0.75rem' }}>
                <div><span style={{ color: '#64748b' }}>Score: </span><span style={{ fontWeight: 700, color: '#fbbf24' }}>{run.combined?.toFixed(3)}</span></div>
                <div><span style={{ color: '#64748b' }}>Acc: </span><span style={{ fontWeight: 600 }}>{(run.accuracy * 100).toFixed(1)}%</span></div>
                <div><span style={{ color: '#64748b' }}>Speed: </span><span style={{ fontWeight: 600 }}>{run.decode_speed?.toFixed(1)}</span></div>
                {run.bleurt_score !== undefined && !isNaN(run.bleurt_score) && <div><span style={{ color: '#64748b' }}>BLEURT: </span><span style={{ color: '#06b6d4' }}>{run.bleurt_score?.toFixed(3)}</span></div>}
                {run.temperature !== undefined && <div><span style={{ color: '#64748b' }}>Temp: </span>{run.temperature}</div>}
                {run.threads !== undefined && <div><span style={{ color: '#64748b' }}>Threads: </span>{run.threads}</div>}
                {run.repeat_penalty !== undefined && <div><span style={{ color: '#64748b' }}>RepPen: </span>{run.repeat_penalty}</div>}
                {run.top_p !== undefined && <div><span style={{ color: '#64748b' }}>Top-P: </span>{run.top_p}</div>}
                {run.top_k !== undefined && <div><span style={{ color: '#64748b' }}>Top-K: </span>{run.top_k}</div>}
                {run.ctx_size !== undefined && <div><span style={{ color: '#64748b' }}>Ctx: </span>{run.ctx_size}</div>}
                {run.batch_size !== undefined && <div><span style={{ color: '#64748b' }}>Batch: </span>{run.batch_size}</div>}
                {run.ubatch_size !== undefined && <div><span style={{ color: '#64748b' }}>UBatch: </span>{run.ubatch_size}</div>}
                {run.flash_attn !== undefined && <div><span style={{ color: '#64748b' }}>Flash: </span>{run.flash_attn}</div>}
                {run.ctk !== undefined && <div><span style={{ color: '#64748b' }}>CTK: </span>{run.ctk}</div>}
                {run.ctv !== undefined && <div><span style={{ color: '#64748b' }}>CTV: </span>{run.ctv}</div>}
                {run.poll_level !== undefined && <div><span style={{ color: '#64748b' }}>Poll: </span>{run.poll_level}</div>}
                {run.use_mmap !== undefined && <div><span style={{ color: '#64748b' }}>MMap: </span>{run.use_mmap}</div>}
                {run.runtime_seconds !== undefined && <div><span style={{ color: '#64748b' }}>Time: </span>{run.runtime_seconds}s</div>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Scatter Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
        {/* Speed vs Accuracy */}
        <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 20 }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#94a3b8' }}>Speed vs Accuracy</h3>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="decode_speed" name="Speed" unit=" tok/s" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis type="number" dataKey="accuracy" name="Accuracy" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
              <Scatter data={filteredData} fill="#3b82f6">
                {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#3b82f6'} fillOpacity={0.8} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* BLEURT vs Accuracy */}
        {hasBleurtData && (
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 20 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#94a3b8' }}>BLEURT vs Accuracy</h3>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="bleurt_score" name="BLEURT" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis type="number" dataKey="accuracy" name="Accuracy" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                <Scatter data={filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score))} fill="#06b6d4">
                  {filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score)).map((entry, index) => <Cell key={`cell-b-${index}`} fill={modelColors[entry.model_short] || '#06b6d4'} fillOpacity={0.8} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Model Combined Score */}
        <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 20 }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#94a3b8' }}>Model Combined Score</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 1]} stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis type="category" dataKey="model" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} width={75} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} />
              <Bar dataKey="combined" name="Combined" radius={[0, 4, 4, 0]}>
                {modelStats.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#3b82f6'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model BLEURT */}
        {hasBleurtData && (
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 16, padding: 20 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#94a3b8' }}>Model BLEURT Scores</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={modelStats.filter(m => m.bleurt !== null)} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis type="category" dataKey="model" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} width={75} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} formatter={v => v?.toFixed(3)} />
                <Bar dataKey="bleurt" name="BLEURT" fill="#06b6d4" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Hyperparameter Impact Charts - Section 1 */}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 16, color: '#10b981' }}>📊 Hyperparameter Impact Analysis</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
        <ParamChart data={tempStats} param="temperature" title="Temperature Impact" />
        <ParamChart data={threadStats} param="threads" title="Thread Count Impact" />
        <ParamChart data={repPenStats} param="repeat_penalty" title="Repeat Penalty Impact" />
        <ParamChart data={topPStats} param="top_p" title="Top-P Impact" />
        <ParamChart data={topKStats} param="top_k" title="Top-K Impact" />
        <ParamChart data={ctxStats} param="ctx_size" title="Context Size Impact" />
        <ParamChart data={batchStats} param="batch_size" title="Batch Size Impact" />
        <ParamChart data={ubatchStats} param="ubatch_size" title="UBatch Size Impact" />
      </div>

      {/* Categorical Parameters */}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 16, color: '#8b5cf6' }}>🔧 Categorical Parameters</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 20, marginBottom: 24 }}>
        <BarParamChart data={flashStats} param="flash_attn" title="Flash Attention" />
        <BarParamChart data={ctkStats} param="ctk" title="KV Cache Key Type (CTK)" />
        <BarParamChart data={ctvStats} param="ctv" title="KV Cache Value Type (CTV)" />
        <BarParamChart data={mmapStats} param="use_mmap" title="Memory Mapping (MMap)" />
        <BarParamChart data={pollStats} param="poll_level" title="Poll Level" />
      </div>

      {/* Summary Stats */}
      <div style={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1))', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: 16, padding: 20, marginBottom: 24 }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 12, color: '#10b981' }}>📈 Summary Statistics</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
          <div>
            <div style={{ color: '#64748b', fontSize: '0.8rem' }}>Total Runs</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{stats?.count}</div>
          </div>
          <div>
            <div style={{ color: '#64748b', fontSize: '0.8rem' }}>Best Accuracy</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#8b5cf6' }}>{(stats?.maxAcc * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div style={{ color: '#64748b', fontSize: '0.8rem' }}>Best Speed</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#10b981' }}>{stats?.maxSpeed.toFixed(1)} tok/s</div>
          </div>
          {stats?.maxBleurt !== null && (
            <div>
              <div style={{ color: '#64748b', fontSize: '0.8rem' }}>Best BLEURT</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#06b6d4' }}>{stats?.maxBleurt.toFixed(3)}</div>
            </div>
          )}
          <div>
            <div style={{ color: '#64748b', fontSize: '0.8rem' }}>Best Combined</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#fbbf24' }}>{stats?.maxCombined.toFixed(3)}</div>
          </div>
          {stats?.avgRuntime !== null && (
            <div>
              <div style={{ color: '#64748b', fontSize: '0.8rem' }}>Avg Runtime</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{stats?.avgRuntime.toFixed(0)}s</div>
            </div>
          )}
        </div>
        
        {hasBleurtData && (
          <div style={{ marginTop: 12, padding: '10px 14px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: 8, border: '1px solid rgba(6, 182, 212, 0.3)', fontSize: '0.8rem' }}>
            <span style={{ color: '#06b6d4', fontWeight: 600 }}>ℹ️ Combined Score:</span>
            <span style={{ color: '#94a3b8', marginLeft: 8 }}>(norm_accuracy + norm_speed + norm_bleurt) / 3</span>
          </div>
        )}
      </div>

      <div style={{ textAlign: 'center', color: '#475569', fontSize: '0.8rem' }}>
        LLM Hyperparameter Analyzer • {filteredData.length} runs analyzed
      </div>
    </div>
  );
}

export default App;

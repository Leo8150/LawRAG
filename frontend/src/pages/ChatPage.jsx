import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import { Send, Activity, ChevronDown, ChevronUp, Download, Save, Clock, FileText } from 'lucide-react'
import { sendChat, clearConversationMemory, getSystemInfo, saveChatRecord, listChatRecords, getChatRecordDownloadUrl } from '../services/api'

export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [useRerank, setUseRerank] = useState(true)
  const [useRewrite, setUseRewrite] = useState(false)
  const [collection, setCollection] = useState('all')
  const [enableMonitor, setEnableMonitor] = useState(true)
  const [enableQuality, setEnableQuality] = useState(false)

  // 高级管线选项
  const [queryTransform, setQueryTransform] = useState('none')
  const [rerankStrategy, setRerankStrategy] = useState('simple')
  const [generationStrategy, setGenerationStrategy] = useState('standard')
  const [useKg, setUseKg] = useState(false)
  const [skillName, setSkillName] = useState('auto')
  const conversationIdRef = useRef(
    globalThis.crypto?.randomUUID?.() || `lawrag-${Date.now()}`
  )

  // 实时监控
  const [liveMonitor, setLiveMonitor] = useState(null)
  const monitorTimerRef = useRef(null)

  // 记录历史
  const [records, setRecords] = useState([])
  const [showRecords, setShowRecords] = useState(false)

  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, liveMonitor])

  // 开始实时监控
  const startLiveMonitor = useCallback(() => {
    const poll = async () => {
      try {
        const info = await getSystemInfo()
        setLiveMonitor(info)
      } catch { /* ignore */ }
    }
    poll()
    monitorTimerRef.current = setInterval(poll, 2000)
  }, [])

  // 停止实时监控
  const stopLiveMonitor = useCallback(() => {
    if (monitorTimerRef.current) {
      clearInterval(monitorTimerRef.current)
      monitorTimerRef.current = null
    }
    setLiveMonitor(null)
  }, [])

  useEffect(() => {
    return () => stopLiveMonitor()
  }, [stopLiveMonitor])

  // 加载记录列表
  const refreshRecords = useCallback(async () => {
    try {
      const list = await listChatRecords()
      setRecords(list || [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    refreshRecords()
  }, [refreshRecords])

  const handleSend = async () => {
    const q = input.trim()
    if (!q || loading) return

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setInput('')
    setLoading(true)

    // 开启实时监控
    if (enableMonitor) {
      startLiveMonitor()
    }

    try {
      const result = await sendChat({
        question: q,
        useRerank,
        useQueryRewrite: useRewrite,
        collection,
        evaluateQuality: enableQuality,
        monitorSystem: enableMonitor,
        queryTransform,
        rerankStrategy,
        generationStrategy,
        useKg,
        conversationId: conversationIdRef.current,
        skillName,
      })

      // 停止实时监控
      stopLiveMonitor()

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.answer,
        sources: result.sources,
        metrics: result.metrics,
        rewrittenQueries: result.rewritten_queries,
        systemBefore: result.system_before,
        systemAfter: result.system_after,
        quality: result.quality,
        question: q,
        kgEntities: result.kg_entities,
        generationStrategy: result.generation_strategy,
        pipelineConfig: result.pipeline_config,
        activeSkill: result.active_skill,
        contextCompaction: result.context_compaction,
      }])
    } catch (err) {
      stopLiveMonitor()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `请求失败: ${err.response?.data?.detail || err.message}`,
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewConversation = async () => {
    const previousId = conversationIdRef.current
    conversationIdRef.current = globalThis.crypto?.randomUUID?.() || `lawrag-${Date.now()}`
    setMessages([])
    try {
      await clearConversationMemory(previousId)
    } catch { /* 新会话已经完成，旧会话交给 TTL 回收 */ }
  }

  // 保存单条记录
  const handleSaveRecord = async (msg) => {
    try {
      const record = {
        question: msg.question || '',
        answer: msg.content,
        metrics: msg.metrics,
        system_before: msg.systemBefore,
        system_after: msg.systemAfter,
        quality: msg.quality,
        sources_count: msg.sources?.length || 0,
      }
      await saveChatRecord(record)
      await refreshRecords()
    } catch (err) {
      alert(`保存失败: ${err.message}`)
    }
  }

  return (
    <div className="chat-container">
      <div className="options-bar">
        <div className="option-group">
          <span className="option-group-label">检索配置</span>
          <label className="option-label" title="查询变换策略">
            <select className="select-small" value={queryTransform} onChange={e => setQueryTransform(e.target.value)}>
              <option value="none">无变换</option>
              <option value="multi_query">多查询扩展</option>
              <option value="hyde">HyDE</option>
              <option value="decompose">子问题分解</option>
              <option value="multi_query_hyde">多查询+HyDE</option>
            </select>
          </label>
          <label className="option-label" title="重排策略">
            <select className="select-small" value={rerankStrategy} onChange={e => setRerankStrategy(e.target.value)}>
              <option value="none">不重排</option>
              <option value="simple">轻量重排</option>
              <option value="cloud">云端重排（消耗额度）</option>
            </select>
          </label>
          {rerankStrategy === 'cloud' && (
            <span className="option-hint" title="调用阿里云百炼 qwen3.7-text-rerank">
              本次问答将调用云端 Reranker
            </span>
          )}
          <select className="select-small" value={collection} onChange={e => setCollection(e.target.value)}>
            <option value="all">全部</option>
            <option value="laws">法律条文</option>
            <option value="cases">指导案例</option>
          </select>
          <label className="option-label">
            <input type="checkbox" checked={useKg} onChange={e => setUseKg(e.target.checked)} />
            KG增强
          </label>
          <select className="select-small" value={skillName} onChange={e => setSkillName(e.target.value)} title="按需加载法律领域 Skill">
            <option value="auto">Skill 自动路由</option>
            <option value="general_legal">通用法律</option>
            <option value="criminal_law">刑事法律</option>
            <option value="labor_law">劳动法律</option>
            <option value="contract_law">合同法律</option>
            <option value="traffic_law">交通法律</option>
          </select>
          <button className="btn-sm" onClick={handleNewConversation} disabled={loading}>新会话</button>
        </div>
        <div className="option-group">
          <span className="option-group-label">生成配置</span>
          <label className="option-label" title="生成策略">
            <select className="select-small" value={generationStrategy} onChange={e => setGenerationStrategy(e.target.value)}>
              <option value="standard">标准</option>
              <option value="structured_legal">结构化法律回答</option>
              <option value="self_reflect">自我修正</option>
            </select>
          </label>
        </div>
        <div style={{ borderLeft: '1px solid var(--border)', height: 20, margin: '0 4px' }} />
        <label className="option-label" style={{ color: enableMonitor ? 'var(--primary)' : undefined }}>
          <input type="checkbox" checked={enableMonitor} onChange={e => setEnableMonitor(e.target.checked)} />
          <Activity size={13} /> 性能监控
        </label>
        <label className="option-label" style={{ color: enableQuality ? '#8b5cf6' : undefined }}>
          <input type="checkbox" checked={enableQuality} onChange={e => setEnableQuality(e.target.checked)} />
          RAG 评测
        </label>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="icon">&#9878;</div>
            <p>请输入法律问题，开始智能问答</p>
            <p style={{ fontSize: 12, marginTop: 8, color: 'var(--text-muted)' }}>
              开启"性能监控"可实时查看系统状态，开启"RAG 评测"可计算检索指标与忠实度
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-bubble">
              {msg.role === 'assistant' ? (
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              ) : (
                msg.content
              )}

              {/* 生成策略标签 + 修正徽章 */}
              {msg.role === 'assistant' && (msg.generationStrategy || msg.metrics?.was_corrected) && (
                <div className="pipeline-badges">
                  {msg.generationStrategy && msg.generationStrategy !== 'standard' && (
                    <span className="badge badge-strategy">
                      {{'self_reflect': '自我修正', 'structured_legal': '结构化'}[msg.generationStrategy] || msg.generationStrategy}
                    </span>
                  )}
                  {msg.metrics?.was_corrected && (
                    <span className="badge badge-corrected">已修正</span>
                  )}
                  {msg.activeSkill && (
                    <span className="badge badge-strategy">Skill: {msg.activeSkill}</span>
                  )}
                </div>
              )}

              {/* KG 实体标签 */}
              {msg.kgEntities?.length > 0 && (
                <div className="kg-entities">
                  <span className="kg-label">KG匹配:</span>
                  {msg.kgEntities.map((e, j) => (
                    <span key={j} className="kg-tag">{e}</span>
                  ))}
                </div>
              )}

              {msg.sources?.length > 0 && (
                <div className="sources-section">
                  <h4>参考来源 ({msg.sources.length})</h4>
                  {msg.sources.map((src, j) => (
                    <div key={j} className="source-card">
                      <div className="source-meta">
                        {src.metadata?.law_name || src.metadata?.guiding_number || src.metadata?.source_file || '来源'}
                        {src.metadata?.article_number && ` 第${src.metadata.article_number}条`}
                        {src.score != null && ` · 相关性 ${(src.score * 100).toFixed(1)}%`}
                      </div>
                      <div className="source-text">{src.content}</div>
                    </div>
                  ))}
                </div>
              )}

              {msg.metrics && (
                <div className="metrics-bar">
                  {msg.metrics.query_rewrite_ms != null && <span className="metric-tag">查询变换 {msg.metrics.query_rewrite_ms}ms</span>}
                  <span className="metric-tag">检索 {msg.metrics.retrieval_ms}ms</span>
                  {msg.metrics.kg_lookup_ms != null && <span className="metric-tag">KG {msg.metrics.kg_lookup_ms}ms</span>}
                  {msg.metrics.rerank_ms != null && <span className="metric-tag">重排序 {msg.metrics.rerank_ms}ms</span>}
                  {msg.metrics.reranker_model && <span className="metric-tag">{msg.metrics.reranker_model}</span>}
                  {msg.metrics.rerank_tokens > 0 && <span className="metric-tag">重排 {msg.metrics.rerank_tokens} tokens</span>}
                  {msg.metrics.rerank_fallback && <span className="metric-tag">重排已降级</span>}
                  <span className="metric-tag">上下文 {msg.metrics.context_tokens_before}→{msg.metrics.context_tokens_after} tokens</span>
                  <span className="metric-tag">Memory {msg.metrics.memory_turns} 轮</span>
                  <span className="metric-tag">Child {msg.metrics.retrieved_child_count}→{msg.metrics.reranked_child_count}</span>
                  <span className="metric-tag">Parent {msg.metrics.parent_candidate_count}</span>
                  <span className="metric-tag">生成 {msg.metrics.generation_ms}ms</span>
                  {msg.metrics.self_reflect_ms != null && <span className="metric-tag">反思 {msg.metrics.self_reflect_ms}ms</span>}
                  <span className="metric-tag">总计 {msg.metrics.total_ms}ms</span>
                </div>
              )}

              {/* 内联性能监控面板 */}
              {msg.role === 'assistant' && (msg.systemBefore || msg.quality) && (
                <MonitorPanel msg={msg} onSave={() => handleSaveRecord(msg)} />
              )}
            </div>
          </div>
        ))}

        {/* 加载中 + 实时监控 */}
        {loading && (
          <div className="message assistant">
            <div className="message-bubble">
              <span className="spinner" /> 正在思考<span className="loading-dots"></span>
              {liveMonitor && (
                <div className="live-monitor-bar" style={{ marginTop: 10 }}>
                  <Activity size={14} style={{ color: 'var(--primary)' }} />
                  <div className="monitor-item">
                    <span className="label">CPU</span>
                    <span className="value">{liveMonitor.cpu_percent}%</span>
                  </div>
                  <div className="monitor-item">
                    <span className="label">内存</span>
                    <span className="value">{liveMonitor.memory_percent}%</span>
                  </div>
                  <div className="monitor-item">
                    <span className="label">已用</span>
                    <span className="value">{liveMonitor.memory_used_gb}GB</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* 记录历史折叠面板 */}
      {records.length > 0 && (
        <div className="records-panel">
          <button className="records-toggle" onClick={() => setShowRecords(v => !v)}>
            <Clock size={14} />
            性能记录 ({records.length})
            {showRecords ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {showRecords && (
            <div style={{ maxHeight: 160, overflowY: 'auto', marginTop: 6 }}>
              {records.map(r => (
                <div key={r.record_id} className="file-item" style={{ marginBottom: 4, padding: '6px 10px' }}>
                  <div className="file-info" style={{ flex: 1 }}>
                    <span className="file-name" style={{ fontSize: 12 }}>{r.question || '...'}</span>
                    <span className="file-meta" style={{ fontSize: 11 }}>
                      {r.created_at?.replace('T', ' ').slice(0, 19)} | {r.total_ms}ms
                      {r.has_quality && ' | 含 RAG 评测'}
                    </span>
                  </div>
                  <a
                    href={getChatRecordDownloadUrl(r.record_id)}
                    download
                    className="btn-sm"
                    style={{ fontSize: 11, padding: '2px 6px', border: '1px solid var(--border)', borderRadius: 4, textDecoration: 'none', color: 'var(--text-muted)' }}
                  >
                    <Download size={10} />
                  </a>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="chat-input-bar">
        <input
          className="input"
          placeholder="请输入法律问题..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button className="btn btn-primary" onClick={handleSend} disabled={loading || !input.trim()}>
          <Send size={16} /> 发送
        </button>
      </div>
    </div>
  )
}


// ===================== 内联监控面板子组件 =====================

function MonitorPanel({ msg, onSave }) {
  const [expanded, setExpanded] = useState(false)
  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave()
    } finally {
      setSaving(false)
    }
  }

  const hasSystem = msg.systemBefore && msg.systemAfter
  const hasQuality = msg.quality

  return (
    <div className="monitor-panel">
      <div className="monitor-header" onClick={() => setExpanded(v => !v)}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Activity size={12} />
          性能 & 质量详情
          {hasQuality && msg.quality.faithfulness && (
            <span style={{ marginLeft: 8, fontSize: 11, color: '#22c55e' }}>
              忠实度 {msg.quality.faithfulness.score}/10
            </span>
          )}
          {hasQuality && msg.quality.retrieval && (
            <span style={{ marginLeft: 8, fontSize: 11, color: '#3b82f6' }}>
              Recall@5 {msg.quality.retrieval.recall_at_5?.toFixed(4)}
            </span>
          )}
        </span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>

      {expanded && (
        <div className="monitor-body">
          {/* 系统前后对比 */}
          {hasSystem && (
            <div className="sys-compare">
              <div className="sys-compare-col">
                <h5>问答前</h5>
                <SysInfoMini info={msg.systemBefore} />
              </div>
              <div className="sys-compare-col">
                <h5>问答后</h5>
                <SysInfoMini info={msg.systemAfter} />
              </div>
            </div>
          )}

          {/* RAG 评测详情 */}
          {hasQuality && (
            <div className="quality-section">
              <h5>RAG 评测</h5>
              <div className="monitor-grid">
                {msg.quality.retrieval && (
                  <>
                    <div className="monitor-card" style={{ borderLeft: '3px solid #8b5cf6' }}>
                      <div className="monitor-value">{msg.quality.retrieval.recall_at_5?.toFixed(4)}</div>
                      <div className="monitor-label">Recall@5</div>
                    </div>
                    <div className="monitor-card" style={{ borderLeft: '3px solid #a78bfa' }}>
                      <div className="monitor-value">{msg.quality.retrieval.mrr_at_10?.toFixed(4)}</div>
                      <div className="monitor-label">MRR@10</div>
                    </div>
                  </>
                )}
                {msg.quality.faithfulness && (
                  <div className="monitor-card" style={{ borderLeft: '3px solid #22c55e' }}>
                    <div className="monitor-value">{msg.quality.faithfulness.score}</div>
                    <div className="monitor-label">忠实度</div>
                  </div>
                )}
              </div>
              {msg.quality.faithfulness?.explanation && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  {msg.quality.faithfulness.explanation}
                </div>
              )}
            </div>
          )}

          {/* 操作按钮 */}
          <div className="monitor-actions">
            <button className="btn-sm" onClick={handleSave} disabled={saving}>
              <Save size={11} /> {saving ? '保存中...' : '保存记录'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}


function SysInfoMini({ info }) {
  if (!info) return null
  return (
    <div>
      <div className="sys-compare-row">
        <span className="label">CPU</span>
        <span className="value">{info.cpu_percent}%</span>
      </div>
      <div className="sys-compare-row">
        <span className="label">内存</span>
        <span className="value">{info.memory_percent}%</span>
      </div>
      <div className="sys-compare-row">
        <span className="label">已用</span>
        <span className="value">{info.memory_used_gb}GB</span>
      </div>
    </div>
  )
}

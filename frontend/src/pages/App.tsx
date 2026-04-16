import { ChangeEvent, FormEvent, useMemo, useState } from 'react'
import { api, setAuthToken } from '../api/client'

type DocItem = { id: string; name: string; metadata?: Record<string, string> }

export function App() {
  const [email, setEmail] = useState('admin@admin.com')
  const [password, setPassword] = useState('admin123')
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)

  const [query, setQuery] = useState('')
  const [response, setResponse] = useState('')
  const [category, setCategory] = useState('general')
  const [message, setMessage] = useState('')
  const [messageType, setMessageType] = useState<'success' | 'error' | ''>('')

  const [documents, setDocuments] = useState<DocItem[]>([])
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])

  const selectedDocNames = useMemo(
    () => documents.filter((d) => selectedDocs.includes(d.id)).map((d) => d.name),
    [documents, selectedDocs]
  )

  const showMessage = (text: string, type: 'success' | 'error') => {
    setMessage(text)
    setMessageType(type)
  }

  const loadDocuments = async () => {
    const res = await api.get('/documents')
    setDocuments(res.data)
  }

  const tryLogin = async () => {
    const res = await api.post('/auth/login', { email, password })
    setAuthToken(res.data.access_token)
    setIsLoggedIn(true)
    await loadDocuments()
    showMessage('Login successful.', 'success')
  }

  const login = async () => {
    try {
      await tryLogin()
    } catch (error) {
      try {
        await api.post('/auth/bootstrap-admin')
        await tryLogin()
        showMessage('Admin bootstrapped and login successful.', 'success')
      } catch (retryError: unknown) {
        console.error(retryError)
        const message =
          typeof retryError === 'object' && retryError && 'response' in retryError
            ? String((retryError as { response?: { data?: { detail?: string } } }).response?.data?.detail || 'Login failed. Check credentials.')
            : 'Login failed. Check email/password.'
        showMessage(message, 'error')
      }
    }
  }

  const onDocSelectionChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const selected = Array.from(event.target.selectedOptions).map((opt) => opt.value)
    setSelectedDocs(selected)
  }

  const upload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!isLoggedIn) {
      showMessage('Please login first.', 'error')
      return
    }

    const form = new FormData(event.currentTarget)
    if (!form.get('file')) {
      showMessage('Please select a PDF file.', 'error')
      return
    }

    try {
      setUploading(true)
      form.append('category', category)
      await api.post('/documents/upload', form)
      await loadDocuments()
      showMessage('File uploaded and indexed.', 'success')
      ;(event.target as HTMLFormElement).reset()
    } catch (error) {
      console.error(error)
      showMessage('Upload failed. Check backend logs.', 'error')
    } finally {
      setUploading(false)
    }
  }

  const streamSSEFromResponse = async (res: Response) => {
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || `HTTP ${res.status}`)
    }

    const reader = res.body?.getReader()
    if (!reader) throw new Error('Stream reader not available')

    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() || ''

      for (const evt of events) {
        const lines = evt.split('\n')
        let data = ''
        for (const line of lines) {
          if (line.startsWith('data:')) data += line.slice(5).trimStart()
        }

        if (!data || data === '[DONE]') continue
        setResponse((prev) => prev + data)
      }
    }
  }

  const ask = async () => {
    if (!isLoggedIn) {
      showMessage('Please login first.', 'error')
      return
    }

    if (!query.trim()) {
      showMessage('Please enter a query.', 'error')
      return
    }

    try {
      setAsking(true)
      setResponse('')
      showMessage('Fetching response...', 'success')

      const authHeader = api.defaults.headers.common.Authorization as string
      const res = await fetch(`${api.defaults.baseURL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authHeader,
        },
        body: JSON.stringify({
          query,
          top_k: 5,
          document_ids: selectedDocs.length ? selectedDocs : undefined,
          filters: category ? { category } : undefined,
        }),
      })

      await streamSSEFromResponse(res)
      showMessage('Response complete.', 'success')
    } catch (error) {
      console.error(error)
      showMessage('Query failed. Check backend/LLM service.', 'error')
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className='container'>
      <h1>Multi-PDF RAG Assistant</h1>

      <div className='card'>
        <h3>1) Login</h3>
        <div className='row'>
          <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder='Email' />
          <input value={password} onChange={(e) => setPassword(e.target.value)} type='password' placeholder='Password' />
          <button onClick={login}>Login</button>
        </div>
      </div>

      <div className='card'>
        <h3>2) Upload PDF</h3>
        <form onSubmit={upload} className='file-row'>
          <input name='file' type='file' accept='application/pdf' required />
          <input value={category} onChange={(e) => setCategory(e.target.value)} placeholder='Metadata category' />
          <button type='submit' disabled={uploading}>{uploading ? 'Uploading...' : 'Upload & Index'}</button>
        </form>
        <p className='muted'>Tip: upload multiple PDFs and then select specific documents for retrieval in step 3.</p>
      </div>

      <div className='card'>
        <h3>3) Ask Question</h3>
        <label className='muted'>Select documents (optional multi-select):</label>
        <select multiple value={selectedDocs} onChange={onDocSelectionChange} style={{ width: '100%', minHeight: 110, marginTop: 8 }}>
          {documents.map((doc) => (
            <option key={doc.id} value={doc.id}>{doc.name}</option>
          ))}
        </select>

        <div className='docs'>
          {selectedDocNames.map((name) => (
            <span className='doc-pill' key={name}>{name}</span>
          ))}
        </div>

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={4}
          style={{ width: '100%', marginTop: 12 }}
          placeholder='Ask your question from uploaded PDFs...'
        />
        <button onClick={ask} disabled={asking} style={{ marginTop: 10 }}>{asking ? 'Generating...' : 'Ask'}</button>
      </div>

      <div className='card'>
        <h3>LLM Output</h3>
        {message && <p className={messageType === 'error' ? 'status-error' : 'status-success'}>{message}</p>}
        <div className='output'>{response || 'Waiting for query...'}</div>
      </div>
    </div>
  )
}

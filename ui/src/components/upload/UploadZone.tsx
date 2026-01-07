import { useState, useCallback, useRef } from 'react'
import { clsx } from 'clsx'
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { uploadApi } from '../../api/upload'
import { Select } from '../common/Select'

interface FileWithStatus {
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  progress: number
  progressMessage?: string  // Current step message
  error?: string
  chunks?: number
}

export function UploadZone() {
  const { collections, activeCollection, setActiveCollection, updateCollection } = useCollectionStore()
  const [files, setFiles] = useState<FileWithStatus[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const selectedCollection = activeCollection || ''

  const acceptedTypes = ['.pdf', '.txt', '.md', '.docx']

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFiles = Array.from(e.dataTransfer.files).filter((file) => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      return acceptedTypes.includes(ext)
    })

    addFiles(droppedFiles)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files))
    }
  }

  const addFiles = (newFiles: File[]) => {
    const fileStatuses: FileWithStatus[] = newFiles.map((file) => ({
      file,
      status: 'pending',
      progress: 0,
    }))
    setFiles((prev) => [...prev, ...fileStatuses])
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (!selectedCollection || files.length === 0) return

    setIsUploading(true)

    const pendingFiles = files.filter((f) => f.status === 'pending')

    for (let i = 0; i < pendingFiles.length; i++) {
      const fileStatus = pendingFiles[i]
      const fileIndex = files.findIndex((f) => f.file === fileStatus.file)

      // Update status to uploading
      setFiles((prev) =>
        prev.map((f, idx) =>
          idx === fileIndex ? { ...f, status: 'uploading', progress: 0, progressMessage: 'Starting upload...' } : f
        )
      )

      try {
        // Use streaming upload with progress updates
        const response = await uploadApi.uploadFileStream(
          {
            files: [fileStatus.file],
            collection: selectedCollection,
          },
          (progress) => {
            // Update progress in real-time
            setFiles((prev) =>
              prev.map((f, idx) =>
                idx === fileIndex
                  ? { ...f, progress: progress.percent, progressMessage: progress.message }
                  : f
              )
            )
          }
        )

        // Update status to success
        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === fileIndex
              ? { ...f, status: 'success', progress: 100, chunks: response.chunks_created, progressMessage: undefined }
              : f
          )
        )

        // Update collection stats
        const collection = collections.find((c) => c.name === selectedCollection)
        if (collection) {
          updateCollection(selectedCollection, {
            document_count: collection.document_count + 1,
            chunk_count: collection.chunk_count + response.chunks_created,
          })
        }
      } catch (error) {
        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === fileIndex
              ? { ...f, status: 'error', error: error instanceof Error ? error.message : 'Upload failed', progressMessage: undefined }
              : f
          )
        )
      }
    }

    setIsUploading(false)
  }

  const pendingCount = files.filter((f) => f.status === 'pending').length
  const successCount = files.filter((f) => f.status === 'success').length

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Upload Documents</h1>
          <p className="text-muted text-sm mt-1">
            Upload files to add them to your collection for RAG queries
          </p>
        </div>

        {/* Collection selector */}
        <Select
          label="Target Collection"
          value={selectedCollection}
          onChange={(val) => setActiveCollection(val || null)}
          options={collections
            .filter((c) => c && c.name && c.name.trim() !== '')
            .map((c) => ({ value: c.name, label: c.name }))}
          placeholder="Select a collection..."
        />

        {/* Drop zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={clsx(
            'border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors',
            isDragging
              ? 'border-accent bg-accent-light'
              : 'border-border hover:border-accent hover:bg-surface-hover'
          )}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={acceptedTypes.join(',')}
            onChange={handleFileSelect}
            className="hidden"
          />
          <div className="w-16 h-16 rounded-2xl bg-surface-hover flex items-center justify-center mx-auto mb-4">
            <Upload className={clsx('h-8 w-8', isDragging ? 'text-accent' : 'text-muted')} />
          </div>
          <p className="text-foreground font-medium mb-1">
            {isDragging ? 'Drop files here' : 'Drop files here or click to browse'}
          </p>
          <p className="text-sm text-muted">
            Supports PDF, TXT, MD, DOCX
          </p>
        </div>

        {/* File list */}
        {files.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-foreground">
                Files ({files.length})
              </h3>
              {pendingCount > 0 && (
                <button
                  onClick={handleUpload}
                  disabled={!selectedCollection || isUploading}
                  className="btn btn-primary btn-sm"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      Upload {pendingCount} file{pendingCount > 1 ? 's' : ''}
                    </>
                  )}
                </button>
              )}
            </div>

            <div className="space-y-2">
              {files.map((fileStatus, index) => (
                <FileItem
                  key={`${fileStatus.file.name}-${index}`}
                  fileStatus={fileStatus}
                  onRemove={() => removeFile(index)}
                />
              ))}
            </div>

            {successCount > 0 && (
              <p className="text-sm text-success flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                {successCount} file{successCount > 1 ? 's' : ''} uploaded successfully
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

interface FileItemProps {
  fileStatus: FileWithStatus
  onRemove: () => void
}

function FileItem({ fileStatus, onRemove }: FileItemProps) {
  const { file, status, progress, progressMessage, error, chunks } = fileStatus

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg border border-border bg-surface">
      <div className="w-10 h-10 rounded-lg bg-surface-hover flex items-center justify-center flex-shrink-0">
        <FileText className="h-5 w-5 text-muted" />
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span>{formatSize(file.size)}</span>
          {status === 'uploading' && progressMessage && (
            <span className="text-accent">{progressMessage}</span>
          )}
          {status === 'success' && chunks && (
            <span className="text-success">{chunks} chunks created</span>
          )}
          {status === 'error' && (
            <span className="text-error">{error}</span>
          )}
        </div>

        {status === 'uploading' && (
          <div className="mt-1.5 h-1.5 rounded-full bg-surface-active overflow-hidden">
            <div
              className="h-full bg-accent transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>

      <div className="flex-shrink-0">
        {status === 'pending' && (
          <button
            onClick={onRemove}
            className="p-1.5 rounded-md text-muted hover:text-foreground hover:bg-surface-hover"
          >
            <X className="h-4 w-4" />
          </button>
        )}
        {status === 'uploading' && (
          <Loader2 className="h-5 w-5 text-accent animate-spin" />
        )}
        {status === 'success' && (
          <CheckCircle className="h-5 w-5 text-success" />
        )}
        {status === 'error' && (
          <AlertCircle className="h-5 w-5 text-error" />
        )}
      </div>
    </div>
  )
}

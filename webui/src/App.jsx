/* lavix-vault-webui v1.0.0 */
import React, { useState, useEffect, useRef, useMemo, Component } from 'react'
import { createPortal } from 'react-dom'
import { api, assistantDisplayText, clearAuthSession, formatApiError, installSessionActivityTracking, renderAssistantMarkdown, renderMarkdown, safeExternalUrl, storeAuthSession, stripAssistantProtocolArtifacts } from './vault-client.js'
import { aiBoxState, documentTypeLabel, fileFormatLabel, followupFileIds,             folderAbsolutePath, getIndexProgress, getIndexState, getIndexStateLabel, getServiceHealthPresentation, isIndexActive, isIndexReady, isIndexSearchable, isIndexUnsupported, isImageFile, isPdfFile, matchesSemanticTags, mergeAdminAndAccountModelConfig, nextSelectionAfterBulkOp, normalizeActiveChatModelConfig, normalizeFollowupSuggestions, normalizeScopeIds, resolveMatchPercentage, resolveMemoryConsent, resolveSendFileIds, unionScopeIds, classifyTagTarget,
            describeIndexingError } from './frontend-workflows.js'
import { mvCanGenerateThumbnail, mvDestroyPdfDocument, mvLoadPdfDocument, mvRenderPdfPage } from './thumbnail-generators.js'



        // ── ICONS ────────────────────────────────────────────────────────────
        // --- ICONS (SVG Components) ---
        const Icons = {
            Shield: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>,
            Upload: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>,
            Bot: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}><path d="M12 8V4H8" /><rect width="16" height="12" x="4" y="8" rx="2" /><path d="M2 14h2" /><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" /></svg>,
            ArrowUp: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>,
            Chat: () => <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>,
            File: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>,
            Folder: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" /></svg>,
            Check: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>,
            AlertTriangle: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>,
            X: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor" className={className}><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>,
            Menu: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor" className={className}><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>,
            MoreHorizontal: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor" className={className}><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" /></svg>,
            Loader: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={`animate-spin ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
            LogOut: ({ size = 20 }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>,
            Zap: ({ size = 20 }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
            Home: ({ size = 20 }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>,
            Trash: ({ size = 20 }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>,
            RotateCw: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
            Grid: ({ size = 20 }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>,
            List: ({ size = 20 }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" /></svg>,
            Download: () => <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>,
            Search: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>,
            Clock: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" strokeWidth="2" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6l4 2" /></svg>,
            Maximize: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 4l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>,
            Minimize: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 9L4 4m0 0h5M4 4l5 5m6-5l5 5m0-5v5m0-5h-5m5 11l-5 5m0-5v5m0-5h5m-11 5l-5-5m0 5v-5m0 5h5" /></svg>,
            Key: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m-5-3a6 6 0 11-7.743 5.743L3 17v2h2v2h2v2h2v-2.257a9.077 9.077 0 012.257-.157L12 21l6-6V9a6 6 0 016-6v2m-9 2v1" /></svg>,
            ChevronLeft: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>,
            ChevronRight: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>,
            MessageSquare: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>,
            Lock: ({ size = 24, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>,
            Paperclip: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>,
            Monitor: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><rect x="2" y="3" width="20" height="14" rx="2" ry="2" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 21h8M12 17v4" /></svg>,
            Sparkles: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3l1.5 4.5L11 9l-4.5 1.5L5 15l-1.5-4.5L-1 9l4.5-1.5z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 3l1 3 3 1-3 1-1 3-1-3-3-1 3-1z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" /></svg>,
            Globe: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" strokeWidth={2}/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>,
            ChevronDown: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>,
            Pencil: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536M9 11l-4 4v4h4l4-4-4-4zm6-6l3.536 3.536L9 17.5 5.5 14 15 5z"/></svg>,
            ArrowUpRight: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 17L17 7M17 7H7M17 7v10"/></svg>,
            Copy: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><rect x="9" y="9" width="13" height="13" rx="2" ry="2" strokeWidth={2}/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>,
            RotateCcw: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><polyline points="1 4 1 10 7 10" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.51 15a9 9 0 1 0 .49-3.75"/></svg>,
            Pin: ({ size = 16, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>,
            Settings: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><circle cx="12" cy="12" r="3" strokeWidth={2} /></svg>,
            Sliders: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}><line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" /><line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" /></svg>,
            User: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>,
            HardDrive: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><polyline points="22 12 2 12" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z"/><line x1="6" y1="16" x2="6.01" y2="16" strokeWidth={2} strokeLinecap="round"/><line x1="10" y1="16" x2="10.01" y2="16" strokeWidth={2} strokeLinecap="round"/></svg>,
            Database: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><ellipse cx="12" cy="5" rx="9" ry="3" strokeWidth={2}/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
            Plus: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v14M5 12h14" /></svg>,
            Info: ({ size = 20, className = "" }) => <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" strokeWidth={2}/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 16v-4m0-4h.01" /></svg>,
        };

        // ── SHARED THEMED DROPDOWN ───────────────────────────────────────
        // Replaces native <select> everywhere: same dashboard tokens
        // (zinc-900, white/10, rounded-xl), portal-escaped so card
        // stacking/overflow can never clip it, full keyboard support.
        // Props: value (string), options [{value,label}|string],
        // onChange(value), disabled, ariaLabel/title, testid, menuMaxHeight.
        function VaultDropdown({ value, options, onChange, disabled, ariaLabel, title, testid, menuMaxHeight = 240 }) {
            const [open, setOpen] = React.useState(false);
            const [highlight, setHighlight] = React.useState(-1);
            const [pos, setPos] = React.useState(null);
            const btnRef = React.useRef(null);
            const menuRef = React.useRef(null);
            const items = React.useMemo(
                () => (options || []).map(opt => typeof opt === 'string' ? { value: opt, label: opt } : opt),
                [options],
            );
            const selected = items.find(item => String(item.value) === String(value ?? ''));
            const openMenu = () => {
                if (disabled || !btnRef.current) return;
                const rect = btnRef.current.getBoundingClientRect();
                const height = Math.min(menuMaxHeight, 36 + items.length * 36);
                const below = window.innerHeight - rect.bottom - 8;
                const above = rect.top - 8;
                const placeAbove = below < Math.min(height, 160) && above > below;
                const width = Math.max(rect.width, 180);
                setPos({
                    top: placeAbove ? Math.max(8, rect.top - Math.min(height, above) - 8) : rect.bottom + 6,
                    left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)),
                    width,
                    maxHeight: placeAbove ? Math.min(height, above) : Math.min(height, below),
                });
                setHighlight(Math.max(0, items.findIndex(item => String(item.value) === String(value ?? ''))));
                setOpen(true);
            };
            const closeMenu = () => { setOpen(false); setHighlight(-1); };
            const choose = (itemValue) => { closeMenu(); btnRef.current?.focus(); onChange && onChange(itemValue); };
            React.useEffect(() => {
                if (!open) return undefined;
                const onDown = (e) => { if (menuRef.current && !menuRef.current.contains(e.target) && e.target !== btnRef.current) closeMenu(); };
                // Capture-phase sees every scroller, including this menu's
                // own overflow list: only dismiss for scrolls that start
                // outside the menu, otherwise the scrollbar is unusable.
                const onScroll = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) closeMenu(); };
                document.addEventListener('mousedown', onDown);
                document.addEventListener('scroll', onScroll, true);
                return () => {
                    document.removeEventListener('mousedown', onDown);
                    document.removeEventListener('scroll', onScroll, true);
                };
            }, [open ]);
            // Keep the highlighted option visible while arrow-keying or
            // hovering a list taller than the menu viewport.
            React.useEffect(() => {
                if (!open || highlight < 0) return;
                menuRef.current?.querySelector(`[role="option"]:nth-child(${highlight + 1})`)?.scrollIntoView({ block: 'nearest' });
            }, [open, highlight, items.length ]);
            const onTriggerKey = (e) => {
                if (disabled) return;
                if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open ? closeMenu() : openMenu(); }
                else if (e.key === 'Escape' && open) { e.preventDefault(); closeMenu(); }
            };
            const onMenuKey = (e) => {
                if (e.key === 'ArrowDown') { e.preventDefault(); setHighlight(h => (h + 1) % Math.max(items.length, 1)); }
                else if (e.key === 'ArrowUp') { e.preventDefault(); setHighlight(h => (h - 1 + items.length) % Math.max(items.length, 1)); }
                else if (e.key === 'Enter') { e.preventDefault(); if (items[highlight]) choose(items[highlight].value); }
                else if (e.key === 'Escape') { e.preventDefault(); closeMenu(); btnRef.current?.focus(); }
            };
            return (
                <>
                    <button
                        ref={btnRef}
                        type="button"
                        disabled={disabled}
                        aria-label={ariaLabel}
                        title={title || (selected ? selected.label : ariaLabel)}
                        aria-haspopup="listbox"
                        aria-expanded={open}
                        data-testid={testid}
                        onClick={() => open ? closeMenu() : openMenu()}
                        onKeyDown={onTriggerKey}
                        className="vault-select flex items-center justify-between gap-2 px-3 py-1.5 min-w-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80 disabled:opacity-50"
                    >
                        <span className="truncate">{selected ? selected.label : (ariaLabel || 'Select…')}</span>
                        <svg width="10" height="10" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" className={`flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}><path d="M6 9l6 6 6-6"/></svg>
                    </button>
                    {open && pos && createPortal(
                        <div
                            ref={menuRef}
                            role="listbox"
                            aria-label={ariaLabel}
                            onKeyDown={onMenuKey}
                            tabIndex={-1}
                            className="fixed z-[9999] bg-zinc-900 border border-white/10 rounded-xl shadow-2xl overflow-y-auto animate-fade-in"
                            style={{ top: pos.top, left: pos.left, width: pos.width, maxHeight: pos.maxHeight }}
                        >
                            {items.length === 0 && <div className="px-4 py-2.5 text-xs text-zinc-600">No options</div>}
                            {items.map((item, index) => {
                                const isSelected = String(item.value) === String(value ?? '');
                                const isHot = index === highlight;
                                return (
                                    <button
                                        key={String(item.value)}
                                        type="button"
                                        role="option"
                                        aria-selected={isSelected}
                                        title={item.label}
                                        onClick={() => choose(item.value)}
                                        onMouseEnter={() => setHighlight(index)}
                                        className={`w-full flex items-center justify-between gap-2 px-4 py-2.5 text-xs transition-colors truncate ${isSelected ? 'text-white bg-white/10' : isHot ? 'text-white bg-white/5' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
                                    >
                                        <span className="truncate">{item.label}</span>
                                        {isSelected && <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" className="flex-shrink-0"><path d="M20 6L9 17l-5-5"/></svg>}
                                    </button>
                                );
                            })}
                        </div>,
                        document.body,
                    )}
                </>
            );
        }

        // ── MARKDOWN RENDERING + API CLIENT ──────────────────────────────────
        // ── MODALS ───────────────────────────────────────────────────────────


        // ── MODALS ───────────────────────────────────────────────────────────
        function ConfirmModal({ title, message, onConfirm, onCancel, confirmText = "Confirm", variant = "danger" }) {
            return (
                <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
                    <div className="bg-[#0f0f10] border border-white/10 w-full max-w-sm rounded-[24px] overflow-hidden shadow-2xl animate-scale-in">
                        <div className="p-6 text-center">
                            <div className={`w-12 h-12 rounded-2xl mx-auto mb-4 flex items-center justify-center ${variant === 'danger' ? 'bg-red-500/10 text-red-500' : 'bg-indigo-500/10 text-indigo-400'}`}>
                                {variant === 'danger' ? <Icons.Trash size={24} /> : <Icons.RotateCw size={24} />}
                            </div>
                            <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
                            <p className="text-zinc-400 text-sm leading-relaxed">{message}</p>
                        </div>
                        <div className="flex border-t border-white/5">
                            <button onClick={onCancel} className="flex-1 px-6 py-4 text-sm font-bold text-zinc-500 hover:text-white hover:bg-white/[0.02] transition-colors border-r border-white/5">
                                No
                            </button>
                            <button onClick={onConfirm} className={`flex-1 px-6 py-4 text-sm font-bold transition-colors ${variant === 'danger' ? 'text-red-400 hover:bg-red-500/5' : 'text-red-400 hover:bg-red-500/5'}`}>
                                {confirmText}
                            </button>
                        </div>
                    </div>
                </div>
            );
        }


        // ── FILE DETAILS PANEL ────────────────────────────────────────────────
        function FileDetails({ file, onClose, onPreview, onDownload, onGrantAccess, onCancelIndexing, onRevokeAccess, onSearch, onChatAboutFile }) {
            if (!file) return null;
            const isFolder = file._type === 'folder';
            const detailFileId = Number(file?.file_id ?? file?.id);
            const isPDF = !isFolder && isPdfFile(file);
            const isSearchable = !isFolder && isIndexSearchable(file);
            const isProcessing = !isFolder && isIndexActive(file);
            const isUnsupported = !isFolder && isIndexUnsupported(file);
            const isLocked = !isFolder && !isSearchable && !isProcessing && !isUnsupported;
            const indexStateLabel = !isFolder ? getIndexStateLabel(file) : '';
            const searchableStatusLabel = isSearchable && !isProcessing ? 'Searchable' : indexStateLabel;
            const intelligenceStatus = String(file.intelligence_status || (file.summary ? 'model' : 'pending')).toLowerCase();
            const intelligenceReady = intelligenceStatus === 'model' || intelligenceStatus === 'fallback';
            const summaryLabel = intelligenceStatus === 'model' ? 'Model summary' : 'Extractive summary';
            // Why the fallback fired (backend metadata intelligibility): shown
            // as a tooltip so extractive output is diagnosable, not silent.
            const intelligenceFallbackHints = {
                invalid_model_response: 'Model output was unusable; showing extractive text',
                ungrounded_summary: 'Model summary did not match the document text',
                evasive_summary: 'Model declined to summarize this document',
                missing_summary: 'Model returned no summary',
                model_timeout: 'Model timed out; showing extractive text',
                model_error: 'Model call failed; showing extractive text',
                vision_description_unavailable: 'Image description unavailable',
                deterministic: 'Deterministic extractive summary',
                intelligence_disabled: 'Document intelligence is disabled',
            };
            const intelligenceFallbackHint = intelligenceStatus === 'fallback'
                ? (intelligenceFallbackHints[String(file.intelligence_fallback_reason || '')] || null)
                : null;
            const formatLabel = !isFolder ? fileFormatLabel(file) : null;
            const [thumbCollapsed, setThumbCollapsed] = React.useState(false);
            const thumbnailRegionId = `pdf-thumbnail-region-${detailFileId}`;
            const panelRef = React.useRef(null);

            // Click-away dismiss: clicking outside the floating panel closes it
            // (same pattern as VaultDropdown). Clicks inside the panel — buttons,
            // preview, scroll — never trigger it.
            React.useEffect(() => {
                if (!file) return undefined;
                const onDocMouseDown = (e) => {
                    if (panelRef.current && !panelRef.current.contains(e.target)) {
                        onClose && onClose();
                    }
                };
                document.addEventListener('mousedown', onDocMouseDown);
                return () => document.removeEventListener('mousedown', onDocMouseDown);
            }, [file, onClose]);

            return (
                <div className="details-floating-panel" ref={panelRef}>
                    {/* ── Panel header ── */}
                    <div className="px-6 pt-6 pb-5 flex items-start justify-between gap-3 flex-shrink-0 border-b border-white/[0.06]">
                        <div className="flex items-start gap-3 min-w-0 flex-1">
                            <div className={`w-11 h-11 rounded-2xl flex-shrink-0 flex items-center justify-center border ${
                                isFolder ? 'bg-indigo-600/10 text-indigo-400 border-indigo-500/20'
                                : isProcessing ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                                : isSearchable ? 'bg-green-600/10 text-green-400 border-green-500/20'
                                : 'bg-zinc-800/80 text-zinc-500 border-white/5'
                            }`}>
                                {isFolder
                                    ? <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                    : isProcessing ? <Icons.Loader size={20} className="animate-spin" />
                                    : isSearchable ? <Icons.Check size={20} />
                                    : <Icons.Lock size={20} />
                                }
                            </div>
                            <div className="min-w-0 flex-1 pt-0.5">
                                <h2 className="text-base font-bold text-white leading-snug line-clamp-2 break-all" title={isFolder ? file.name : file.filename}>
                                    {isFolder ? file.name : file.filename}
                                </h2>
                                <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                                    {isFolder ? (
                                        <span className="text-[10px] font-semibold text-zinc-500 bg-white/[0.03] border border-white/5 px-2 py-0.5 rounded-full">
                                            {file.file_count ?? 0} item{file.file_count !== 1 ? 's' : ''}
                                        </span>
                                    ) : (
                                        <>
                                            {formatLabel && <span className="text-[10px] font-bold text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full tracking-wide">Format: {formatLabel}</span>}
                                            {file.doc_type && <span className="text-[10px] font-bold text-violet-300 bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded-full tracking-wide">File type: {documentTypeLabel(file.doc_type)}</span>}
                                            <span className={`text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full border ${isSearchable ? 'text-green-400 bg-green-500/10 border-green-500/20' : isProcessing ? 'text-amber-400 bg-amber-500/10 border-amber-500/20' : 'text-zinc-500 bg-white/[0.03] border-white/5'}`}>
                                                {searchableStatusLabel}
                                            </span>
                                            <span className="text-[10px] font-mono text-zinc-500 bg-white/[0.03] border border-white/5 px-2 py-0.5 rounded-full flex items-center gap-1.5"
                                                  title="File ID">
                                                ID: {file.id || file.file_id}
                                            </span>
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>
                        <div className="flex items-center gap-0.5 flex-shrink-0">
                            {/* Download — quiet icon in header */}
                            {!isFolder && (
                                <button onClick={() => onDownload(file.id, file.filename)}
                                    className="p-1.5 hover:bg-white/[0.06] rounded-xl text-zinc-600 hover:text-zinc-200 transition-all"
                                    title="Download">
                                    <Icons.Download size={15} />
                                </button>
                            )}
                            {isProcessing && (
                                <button onClick={() => onCancelIndexing && onCancelIndexing(file.id)}
                                    className="p-1.5 hover:bg-amber-500/10 rounded-xl text-zinc-700 hover:text-amber-400 transition-all"
                                    title="Cancel indexing; keep the published index and AI access">
                                    <Icons.X size={14} />
                                </button>
                            )}
                            {isSearchable && !isProcessing && (
                                <button onClick={() => onRevokeAccess && onRevokeAccess(file.id)}
                                    className="p-1.5 hover:bg-red-500/10 rounded-xl text-zinc-700 hover:text-red-400 transition-all"
                                    title="Revoke AI access and remove derived index data">
                                    <Icons.Lock size={14} />
                                </button>
                            )}
                            <button type="button" onClick={onClose} aria-label="Close file details" className="flex h-10 w-10 items-center justify-center rounded-xl text-zinc-500 transition-all hover:bg-white/[0.06] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80">
                                <Icons.X size={16} />
                            </button>
                        </div>
                    </div>

                    {/* ── Scrollable content ── */}
                    <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5 custom-scrollbar">
                        {/* Document preview thumbnail — only for files */}
                        {!isFolder && (
                            <div className="rounded-2xl border border-white/[0.07] bg-[#0c0c0e] overflow-hidden">
                                {/* Preview actions are separate controls: regeneration must not collapse or open the viewer. */}
                                <div className="flex min-h-12 items-center justify-between gap-2 border-b border-white/[0.05] px-3 py-1.5">
                                    <span className="text-[10px] font-black uppercase tracking-[0.12em] text-zinc-500">Preview</span>
                                    <div className="flex items-center gap-1.5">
                                        <button
                                            data-testid="thumbnail-collapse-toggle"
                                            type="button"
                                            onClick={() => setThumbCollapsed(v => !v)}
                                            aria-expanded={!thumbCollapsed}
                                            aria-controls={thumbnailRegionId}
                                            aria-label={thumbCollapsed ? 'Expand preview' : 'Collapse preview'}
                                            className="inline-flex h-10 w-10 items-center justify-center rounded-xl text-zinc-500 transition-all hover:bg-white/[0.06] hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80"
                                        >
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                                                className={`transition-transform ${thumbCollapsed ? 'rotate-180' : ''}`}>
                                                <path d="M18 15l-6-6-6 6"/>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                                {/* Thumbnail — collapses with smooth transition */}
                                <div id={thumbnailRegionId} className={`transition-all duration-300 ease-in-out overflow-hidden ${thumbCollapsed ? 'max-h-0' : 'max-h-[220px]'}`}>
                                    <div className="relative cursor-pointer group/thumb"
                                         onClick={() => { onClose(); onPreview(file); }}>
                                        <FileThumbnail key={`details-thumbnail-${detailFileId}`} file={file} size="panel" />
                                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-all group-hover/thumb:bg-black/30 group-hover/thumb:opacity-100">
                                            <div className="flex items-center gap-1.5 rounded-xl bg-black/70 px-3 py-1.5 text-xs font-bold text-white backdrop-blur-sm">
                                                <Icons.Zap size={12} /> Preview
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        {isFolder ? (
                            <>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-white/[0.025] border border-white/[0.06] rounded-2xl p-4">
                                        <p className="text-[10px] font-black uppercase tracking-[0.15em] text-zinc-600 mb-2">ITEMS</p>
                                        <p className="text-xl font-bold text-zinc-200">{file.file_count ?? 0}</p>
                                    </div>
                                    <div className="bg-white/[0.025] border border-white/[0.06] rounded-2xl p-4">
                                        <p className="text-[10px] font-black uppercase tracking-[0.15em] text-zinc-600 mb-2">CREATED</p>
                                        <p className="text-xs font-bold text-zinc-300 leading-relaxed">{new Date(file.created_at).toLocaleDateString(undefined, {year:'numeric',month:'short',day:'numeric'})}</p>
                                    </div>
                                </div>
                                <div className="bg-white/[0.025] border border-white/[0.06] rounded-2xl p-4">
                                    <p className="text-[10px] font-black uppercase tracking-[0.15em] text-zinc-600 mb-2">FOLDER NAME</p>
                                    <p className="text-sm font-semibold text-zinc-200 break-all">{file.name}</p>
                                </div>
                                <div className="bg-indigo-600/5 border border-indigo-500/15 rounded-2xl p-4 flex items-center gap-3">
                                    <div className="w-9 h-9 rounded-2xl bg-indigo-600/10 flex items-center justify-center flex-shrink-0">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-indigo-300 mb-0.5">Open folder</p>
                                        <p className="text-[10px] text-zinc-600">Click the ⋯ menu on the card, then "Open"</p>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <>
                                {/* Stat chips (STATUS lives in the header badge; this cell is the chat action) */}
                                <div className="grid grid-cols-3 gap-2">
                                    <button
                                        data-testid="chat-about-file"
                                        type="button"
                                        onClick={() => onChatAboutFile && onChatAboutFile(file)}
                                        title="Open a new chat scoped to this file"
                                        className="bg-indigo-600/[0.07] border border-indigo-500/20 hover:bg-indigo-600/[0.14] rounded-2xl p-3.5 text-left transition-all active:scale-[0.98]"
                                    >
                                        <p className="text-[9px] font-black uppercase tracking-[0.15em] text-indigo-400/80 mb-2">Chat</p>
                                        <div className="flex items-center justify-center py-2 min-h-[44px] text-indigo-300">
                                            <Icons.MessageSquare size={24} />
                                        </div>
                                        <p className="text-[10px] text-zinc-500 leading-tight mt-1">Talk about this file</p>
                                    </button>
                                    <div className="bg-white/[0.025] border border-white/[0.06] rounded-2xl p-3.5">
                                        <p className="text-[9px] font-black uppercase tracking-[0.15em] text-zinc-600 mb-2">SIZE</p>
                                        <p className="text-[11px] font-bold text-zinc-200">{formatSize(file.size_bytes)}</p>
                                    </div>
                                    <div className="bg-white/[0.025] border border-white/[0.06] rounded-2xl p-3.5">
                                        <p className="text-[9px] font-black uppercase tracking-[0.15em] text-zinc-600 mb-2">ADDED</p>
                                        <p className="text-[10px] font-bold text-zinc-300 leading-tight">{new Date(file.uploaded_at).toLocaleDateString(undefined,{month:'short',day:'numeric',year:'2-digit'})}</p>
                                    </div>
                                </div>

                                {/* Locked: grant access */}
                                {isLocked && (
                                    <div className="rounded-2xl overflow-hidden border border-white/[0.06] relative">
                                        <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/10 to-transparent pointer-events-none"></div>
                                        <div className="p-6 flex flex-col items-center text-center relative z-10">
                                            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-4">
                                                <Icons.Shield size={22} className="text-indigo-400" />
                                            </div>
                                            <h3 className="text-sm font-bold text-white mb-1.5">Allow Lavix AI Access</h3>
                                            <p className="text-xs text-zinc-400 mb-5 leading-relaxed max-w-[220px]">Analyze this document for smart summaries, semantic tags, and PostgreSQL vector search.</p>
                                            <button onClick={() => onGrantAccess && onGrantAccess(file.id)} className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white font-bold text-xs transition-all active:scale-[0.98]">
                                                <Icons.Shield size={13} /> GIVE AI ACCESS
                                            </button>
                                        </div>
                                    </div>
                                )}

                                {/* Processing state */}
                                {isProcessing && (
                                    <div className="rounded-2xl border border-amber-500/20 px-5 py-4 flex items-center gap-3 bg-amber-500/[0.04]">
                                        <Icons.Loader size={15} className="text-amber-400 animate-spin flex-shrink-0" />
                                        <div>
                                            <p className="text-xs font-bold text-amber-300">{indexStateLabel}</p>
                                            <p className="text-[10px] text-amber-400/60 mt-0.5">Summary &amp; tags will appear when complete</p>
                                        </div>
                                    </div>
                                )}

                                {/* Searchable revision: document intelligence */}
                                {isSearchable && intelligenceReady && (
                                    <>
                                        <div>
                                            <div className="flex items-center justify-between mb-3">
                                                <span className="text-[10px] font-black uppercase tracking-[0.15em] text-zinc-500">AI INTELLIGENCE</span>
                                                <div title={intelligenceFallbackHint || undefined} className={`px-2 py-0.5 rounded-full text-[9px] font-bold border tracking-wide uppercase ${intelligenceStatus === 'model' ? 'bg-indigo-500/10 text-indigo-300 border-indigo-500/20' : 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20'}`}>{summaryLabel}</div>
                                            </div>
                                            <div className="bg-white/[0.025] border border-white/[0.06] p-5 rounded-2xl relative overflow-hidden group">
                                                <div className="absolute top-0 right-0 p-4 opacity-[0.04] group-hover:opacity-[0.07] transition-opacity pointer-events-none">
                                                    <Icons.Shield size={52} />
                                                </div>
                                                <div className="text-[13px] text-zinc-300 leading-relaxed font-medium relative z-10 markdown-content"
                                                    dangerouslySetInnerHTML={{ __html: renderMarkdown((file.summary || "No reliable summary is available.").replace(/\r\n/g, '\n').replace(/([^\n])\n([^\n])/g, '$1 $2').trim()) }} />
                                            </div>
                                        </div>
                                        {/* Semantic tags */}
                                        <div>
                                            <p className="text-[10px] font-black uppercase tracking-[0.15em] text-zinc-500 mb-3">SEMANTIC TAGS</p>
                                            <div className="flex flex-wrap gap-1.5">
                                                {(file.tags || []).length > 0
                                                    ? file.tags.map((tag, i) => (
                                                        <button 
                                                            key={i} 
                                                            onClick={(e) => { 
                                                                e.stopPropagation(); 
                                                                if (onSearch) onSearch('#' + tag.toLowerCase()); 
                                                                onClose(); 
                                                            }}
                                                            className="bg-zinc-900/60 border border-white/[0.06] text-zinc-400 text-[10px] font-bold px-2.5 py-1 rounded-full hover:border-indigo-500/35 hover:text-indigo-400 transition-all cursor-pointer"
                                                            title={`Search for #${tag.toLowerCase()}`}
                                                        >
                                                            #{tag.toLowerCase()}
                                                        </button>
                                                    ))
                                                    : <span className="text-zinc-600 text-[10px] font-medium italic py-1">No reliable semantic tags</span>
                                                }
                                            </div>
                                        </div>
                                    </>
                                )}
                                {isSearchable && !intelligenceReady && !isProcessing && (
                                    <div className="rounded-2xl border border-white/[0.06] px-5 py-4 flex items-center gap-3 bg-white/[0.02]">
                                        <Icons.Loader size={15} className="text-zinc-500 animate-spin flex-shrink-0" />
                                        <div>
                                            <p className="text-xs font-bold text-zinc-300">Document intelligence pending</p>
                                            <p className="text-[10px] text-zinc-600 mt-0.5">The searchable index is ready; summary and tags are still being finalized.</p>
                                        </div>
                                    </div>
                                )}

                                {/* Single primary CTA */}
                                <div className="pt-1">
                                    <button onClick={() => { onClose(); onPreview(file); }}
                                        className="w-full h-11 rounded-2xl bg-indigo-600/20 border border-indigo-500/25 hover:bg-indigo-600/30 flex items-center justify-center gap-2 text-indigo-300 hover:text-indigo-200 transition-all text-xs font-bold tracking-wide active:scale-[0.98]">
                                        <Icons.Zap size={15} /> Open Preview
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            );
        }


        // ── SKELETON COMPONENTS ──────────────────────────────────────────────────
        function SkeletonFileCard() {
            return (
                <div className="glass-card p-4 flex flex-col gap-3">
                    <div className="skeleton w-full rounded-xl" style={{aspectRatio:'16/9'}}></div>
                    <div className="skeleton h-4 w-3/4 rounded"></div>
                    <div className="skeleton h-3 w-1/2 rounded"></div>
                    <div className="flex justify-between mt-auto pt-2">
                        <div className="skeleton h-6 w-16 rounded-full"></div>
                        <div className="skeleton h-6 w-12 rounded"></div>
                    </div>
                </div>
            );
        }
        function SkeletonFolderChip() {
            return (
                <div className="folder-grid-card">
                    <div className="skeleton w-10 h-10 rounded-xl flex-shrink-0"></div>
                    <div className="flex flex-col gap-1.5">
                        <div className="skeleton h-3 w-24 rounded"></div>
                        <div className="skeleton h-2 w-14 rounded"></div>
                    </div>
                </div>
            );
        }

        // ── FOLDER CONTEXT MENU ─────────────────────────────────────────────────
        function FolderContextMenu({ folder, openUpward, anchorRect, onOpen, onRename, onDelete, onDownload, onDetails, onClose }) {
            const ref = React.useRef(null);
            useEffect(() => {
                const handler = (e) => {
                    if (ref.current && !ref.current.contains(e.target)) onClose();
                };
                const scrollHandler = () => onClose();
                document.addEventListener('mousedown', handler);
                // A fixed-position portal cannot track scrolled content; close instead of floating detached.
                document.addEventListener('scroll', scrollHandler, true);
                return () => {
                    document.removeEventListener('mousedown', handler);
                    document.removeEventListener('scroll', scrollHandler, true);
                };
            }, [onClose]);
            const items = [
                { label: 'Open', icon: <Icons.ArrowUpRight size={14}/>, action: onOpen },
                ...(onDetails ? [{ label: 'Details', icon: <Icons.Info size={14}/>, action: onDetails }] : []),
                { label: 'Download', icon: <Icons.Download size={14}/>, action: onDownload },
                { label: 'Rename', icon: <Icons.Pencil size={14}/>, action: onRename },
                { label: 'Delete', icon: <Icons.Trash size={14}/>, action: onDelete, danger: true },
            ];
            // Portal to body: the menu must escape the card's stacking context
            // (hovered sibling cards create their own via translateY) or
            // neighbors paint over it. Fixed position from the trigger rect.
            const MENU_W = 176; // w-44
            const MENU_H = 168; // ~5 items
            const top = openUpward
                ? Math.max(8, (anchorRect?.top ?? 0) - MENU_H - 4)
                : (anchorRect?.bottom ?? 0) + 8;
            const left = Math.max(8, Math.min(
                (anchorRect?.right ?? 0) - MENU_W,
                window.innerWidth - MENU_W - 8,
            ));
            return createPortal(
                <div ref={ref}
                    className="fixed z-[9999] bg-[#111113] border border-white/10 rounded-2xl shadow-2xl shadow-black/60 py-1.5 w-44"
                    style={{ top, left, animation: 'floatIn 0.18s cubic-bezier(0.34,1.56,0.64,1)' }}>
                    {items.map(item => (
                        <button key={item.label}
                            onClick={e => { e.stopPropagation(); item.action(e); onClose(); }}
                            className={`w-full flex items-center gap-2.5 px-3.5 py-2 text-xs font-medium transition-colors hover:bg-white/5 ${item.danger ? 'text-red-400 hover:text-red-300' : 'text-zinc-300 hover:text-white'}`}>
                            {item.icon} {item.label}
                        </button>
                    ))}
                </div>,
                document.body,
            );
        }

        // ── FOLDER CARD ─────────────────────────────────────────────────────────
        function FolderCard({ folder, isSelected, renamingId, renameValue, setRenameValue,
                              onSelect, onOpen, onRename, onRenameConfirm, onRenameCancel, onDelete, onDownload, onDetails }) {
            const [menuOpen, setMenuOpen] = React.useState(false);
            const [openUpward, setOpenUpward] = React.useState(false);
            const [menuAnchor, setMenuAnchor] = React.useState(null);
            const btnRef = React.useRef(null);
            const openMenu = (e) => {
                e.stopPropagation();
                if (btnRef.current) {
                    const rect = btnRef.current.getBoundingClientRect();
                    // 180px = approx menu height (4 items)
                    setOpenUpward(rect.bottom + 180 > window.innerHeight);
                    setMenuAnchor({ top: rect.top, bottom: rect.bottom, right: rect.right });
                }
                setMenuOpen(v => !v);
            };
            return (
                <div className={`folder-grid-card group ${isSelected ? 'selected' : ''}`}
                     onClick={onOpen}
                     style={{ position: 'relative', zIndex: menuOpen ? 50 : 'auto' }}>
                    {/* 3-dot menu — top right, shown on hover */}
                    <div className="absolute top-2 right-2 z-20">
                        <button
                            ref={btnRef}
                            onClick={openMenu}
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-xl hover:bg-white/10 text-zinc-500 hover:text-white transition-all">
                            <Icons.MoreHorizontal size={15} />
                        </button>
                        {menuOpen && (
                            <FolderContextMenu
                                folder={folder}
                                openUpward={openUpward}
                                anchorRect={menuAnchor}
                                onOpen={() => { onOpen(); setMenuOpen(false); }}
                                onDetails={onDetails ? () => { onDetails(); setMenuOpen(false); } : undefined}
                                onDownload={() => { onDownload && onDownload(); setMenuOpen(false); }}
                                onRename={(e) => { onRename(e); setMenuOpen(false); }}
                                onDelete={() => { onDelete(); setMenuOpen(false); }}
                                onClose={() => setMenuOpen(false)} />
                        )}
                    </div>
                    {/* Folder icon */}
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center flex-shrink-0">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400">
                            <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
                        </svg>
                    </div>
                    {/* Name — editable when renaming */}
                    {renamingId === `folder:${folder.id}` ? (
                        <input autoFocus type="text" value={renameValue}
                            onChange={e => setRenameValue(e.target.value)}
                            onClick={e => e.stopPropagation()}
                            onKeyDown={e => {
                                e.stopPropagation();
                                if (e.key === 'Enter') onRenameConfirm();
                                if (e.key === 'Escape') onRenameCancel();
                            }}
                            onBlur={onRenameConfirm}
                            className="bg-white/5 border border-indigo-500/50 rounded-lg px-2 py-0.5 text-xs text-white outline-none w-full" />
                    ) : (
                        <p className="text-xs font-semibold text-white truncate leading-tight pr-6">{folder.name}</p>
                    )}
                    <p className="text-[10px] text-zinc-600">{folder.file_count ?? 0} item{folder.file_count !== 1 ? 's' : ''}{folder.total_size_bytes > 0 ? ` • ${formatSize(folder.total_size_bytes)}` : ''}</p>
                </div>
            );
        }


        // ── WEB VIEWER ──────────────────────────────────────────────────────────
        function WebViewer({ textUrl, onLoad }) {
            const [content, setContent] = useState('');
            const [error, setError] = useState(null);

            useEffect(() => {
                const fetchText = async () => {
                    try {
                        const res = await fetch(textUrl);
                        if (!res.ok) throw new Error('Failed to load content');
                        const text = await res.text();
                        setContent(text);
                        onLoad && onLoad();
                    } catch (e) {
                        setError(e.message);
                        onLoad && onLoad();
                    }
                };
                fetchText();
            }, [textUrl, onLoad]);

            if (error) return <div className="p-8 text-center text-red-400">{error}</div>;
            return <>{content}</>;
        }

        function PdfDocumentViewer({ previewUrl, sizeBytes, onLoad, onError }) {
            const containerRef = useRef(null);
            const pdfRef = useRef(null);
            const [pages, setPages] = useState([]);
            const [loading, setLoading] = useState(true);
            const [loadError, setLoadError] = useState(null);

            useEffect(() => {
                let cancelled = false;
                const controller = new AbortController();
                setPages([]);
                setLoading(true);
                setLoadError(null);
                mvLoadPdfDocument(previewUrl, sizeBytes, { signal: controller.signal })
                    .then(async pdf => {
                        if (cancelled) {
                            mvDestroyPdfDocument(pdf).catch(() => {});
                            return;
                        }
                        pdfRef.current = pdf;
                        const numPages = pdf.numPages;
                        const availableWidth = Math.max(320, (containerRef.current?.clientWidth || 960) - 48);
                        const rendered = [];
                        for (let i = 1; i <= numPages; i++) {
                            if (cancelled) break;
                            const canvas = document.createElement('canvas');
                            try {
                                await mvRenderPdfPage(pdf, i, canvas, { maxWidth: availableWidth, signal: controller.signal });
                                rendered.push({ page: i, src: canvas.toDataURL('image/webp', 0.95), width: canvas.style.width, height: canvas.style.height });
                            } catch (err) {
                                if (!cancelled) rendered.push({ page: i, error: true });
                            }
                        }
                        if (!cancelled) {
                            setPages(rendered);
                            setLoading(false);
                            onLoad();
                        }
                    })
                    .catch(error => {
                        if (!cancelled) {
                            setLoadError(error);
                            setLoading(false);
                            onError(error);
                        }
                    });
                return () => {
                    cancelled = true;
                    controller.abort();
                    const pdf = pdfRef.current;
                    pdfRef.current = null;
                    if (pdf) mvDestroyPdfDocument(pdf).catch(() => {});
                };
            }, [previewUrl, sizeBytes]);

            return (
                <div ref={containerRef} className="relative flex h-full w-full flex-col bg-zinc-900">
                    <div className="sticky top-0 z-10 flex h-12 flex-none items-center justify-center gap-3 border-b border-white/10 bg-black/80 px-4 backdrop-blur">
                        <span data-testid="pdf-viewer-page-status" className="text-xs font-semibold text-zinc-300">{loading ? 'Loading…' : `${pages.length} page${pages.length !== 1 ? 's' : ''}`}</span>
                    </div>
                    <div className="flex-1 overflow-auto p-4">
                        {loadError && (
                            <div className="flex items-center justify-center h-full text-red-400 text-sm">{loadError.message || 'Failed to load PDF'}</div>
                        )}
                        {loading && !loadError && (
                            <div className="flex items-center justify-center h-full text-zinc-500 text-sm">Rendering pages…</div>
                        )}
                        {pages.map(p => (
                            p.error ? (
                                <div key={p.page} className="mx-auto mb-4 flex items-center justify-center h-32 rounded-lg border border-red-500/20 bg-red-500/5 text-red-400 text-xs">
                                    Page {p.page} — render error
                                </div>
                            ) : (
                                <img key={p.page} src={p.src} alt={`Page ${p.page}`} style={{ width: p.width, height: p.height }}
                                    className="mx-auto mb-4 bg-white shadow-2xl rounded-sm" />
                            )
                        ))}
                    </div>
                </div>
            );
        }


        // ── FILE VIEWER ─────────────────────────────────────────────────────────
        function FileViewer({ file, onClose, onDetails }) {
            const [session, setSession] = useState(null);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [downloadError, setDownloadError] = useState(null);

            const sourceFileId = Number(file.file_id);
            const directFileId = Number(file.id);
            const fileId = Number.isInteger(sourceFileId) && sourceFileId > 0
                ? sourceFileId
                : (Number.isInteger(directFileId) && directFileId > 0 ? directFileId : null);
            const fileName = file.filename || file.original_filename || `File ${fileId}`;
            const isPDF = isPdfFile(file);

            const loadSession = async (force = false) => {
                setLoading(true);
                setError(null);
                try {
                    if (!fileId) throw new Error('Invalid file ID');
                    const data = window.mvGetPreviewSession
                        ? await window.mvGetPreviewSession(fileId, api.createPreviewSession, { force })
                        : await api.createPreviewSession(fileId);
                    setSession(data);
                } catch (e) {
                    setSession(null);
                    setError(e?.message || 'Failed to create preview session');
                    setLoading(false);
                }
            };

            useEffect(() => {
                if (!isImage) loadSession();
            }, [fileId]);

            const isText = file.mime_type?.startsWith('text/') || fileName.match(/\.(txt|md|py|js|sh|json|css|html)$/);
            const isImage = file.mime_type?.startsWith('image/') || isImageFile(file);
            const isVideo = file.mime_type?.startsWith('video/') || fileName.match(/\.(mp4|mkv|webm|mov)$/);
            const previewUrl = session?.preview_url || session?.previewUrl;
            const handlePreviewError = React.useCallback((reason) => {
                setSession(null);
                setError(reason?.message || 'Unable to render this preview');
                setLoading(false);
            }, []);
            const handlePreviewLoad = React.useCallback(() => setLoading(false), []);
            const handleDownload = React.useCallback(async () => {
                setDownloadError(null);
                try {
                    await api.downloadFile(fileId, fileName);
                } catch (reason) {
                    setDownloadError(reason?.message || 'Download failed');
                }
            }, [fileId, fileName]);
            const [fullscreen, setFullscreen] = React.useState(false);
            const viewerRef = React.useRef(null);
            const handleFullscreen = React.useCallback(() => {
                if (!document.fullscreenElement) {
                    viewerRef.current?.requestFullscreen?.();
                    setFullscreen(true);
                } else {
                    document.exitFullscreen();
                    setFullscreen(false);
                }
            }, []);
            React.useEffect(() => {
                const onFsChange = () => setFullscreen(!!document.fullscreenElement);
                document.addEventListener('fullscreenchange', onFsChange);
                return () => document.removeEventListener('fullscreenchange', onFsChange);
            }, []);

            // Auto-clear loading for unsupported types
            useEffect(() => {
                if (session && !isPDF && !isImage && !isVideo && !isText) {
                    setLoading(false);
                }
            }, [session, isPDF, isImage, isVideo, isText]);

            // Images bypass the document-session pipeline (built for
            // PDFs/video/text): decrypted bytes render directly.
            const [imgUrl, setImgUrl] = useState(null);
            useEffect(() => {
                let cancelled = false;
                setImgUrl(null);
                if (!isImage || !fileId) return;
                (async () => {
                    try {
                        const url = await api.fetchFileObjectUrl(fileId);
                        if (!cancelled) { setImgUrl(url); setLoading(false); }
                        else api.revokeFileObjectUrl(url);
                    } catch (e) {
                        if (!cancelled) handlePreviewError(e);
                    }
                })();
                return () => { cancelled = true; };
            }, [fileId, isImage]);
            useEffect(() => {
                return () => { if (imgUrl) api.revokeFileObjectUrl(imgUrl); };
            }, [imgUrl]);

            return (
                <div ref={viewerRef} data-testid="file-viewer" role="dialog" aria-modal="true" aria-labelledby="file-viewer-title" className="fixed inset-0 z-[300] flex items-center justify-center p-4 bg-black/90 backdrop-blur-md animate-fade-in">
                    <div className="glass-card glass-card-static w-full max-w-5xl h-[90vh] flex flex-col overflow-hidden border-white/10">
                        <div className="p-4 border-b border-white/10 flex items-center justify-between gap-4 bg-black/40">
                            <div className="flex items-center gap-3 min-w-0 flex-1">
                                <div className="p-2 bg-indigo-600/20 rounded-lg text-indigo-400 flex-shrink-0">
                                    <Icons.File size={20} />
                                </div>
                                <h2 id="file-viewer-title" className="font-bold text-lg line-clamp-2 break-all text-white" title={fileName}>{fileName}</h2>
                            </div>
                            <div className="flex items-center gap-2 flex-shrink-0">
                                <button data-testid="file-viewer-download" type="button" onClick={handleDownload} className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80" title="Download" aria-label="Download file">
                                    <Icons.Download size={20} />
                                </button>
                                {onDetails && (
                                    <button data-testid="file-viewer-details" type="button" onClick={() => onDetails(file)} className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80" title="Details" aria-label="Show file details">
                                        <Icons.Info size={20} />
                                    </button>
                                )}
                                <button data-testid="file-viewer-fullscreen" type="button" onClick={handleFullscreen} className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80" title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'} aria-label={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
                                    {fullscreen ? <Icons.Minimize size={20} /> : <Icons.Maximize size={20} />}
                                </button>
                                <button data-testid="file-viewer-close" type="button" onClick={onClose} className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80" aria-label="Close file viewer">
                                    <Icons.X size={20} />
                                </button>
                            </div>
                        </div>

                        {downloadError && (
                            <div data-testid="file-viewer-download-error" role="alert" className="flex flex-none items-center justify-between gap-3 border-b border-red-500/15 bg-red-500/[0.06] px-4 py-2 text-xs text-red-300">
                                <span>{downloadError}</span>
                                <button data-testid="file-viewer-download-retry" type="button" onClick={handleDownload} className="min-h-10 rounded-lg border border-red-500/25 px-3 font-semibold hover:bg-red-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400/80">Retry download</button>
                            </div>
                        )}

                        <div className="flex-1 overflow-auto bg-[#0a0a0b] flex items-center justify-center relative">
                            {loading && (
                                <div className="flex flex-col items-center gap-3 text-indigo-400">
                                    <div className="animate-spin text-3xl"><Icons.Loader size={32} /></div>
                                    <p className="text-sm font-medium">Decrypting Vault File...</p>
                                </div>
                            )}

                            {error && (
                                <div data-testid="file-viewer-error" role="alert" className="text-center p-8">
                                    <div className="text-red-400 text-4xl mb-4 mx-auto"><Icons.X size={48} /></div>
                                    <p className="text-red-400 font-medium mb-2">{error}</p>
                                    <div className="flex items-center justify-center gap-2 mt-4">
                                        <button data-testid="file-viewer-retry" type="button" onClick={() => loadSession(true)} className="btn-primary min-h-10 rounded-xl px-6">Retry</button>
                                        <button type="button" onClick={onClose} className="min-h-10 rounded-xl border border-white/10 px-6 text-zinc-400 hover:text-white">Close</button>
                                    </div>
                                </div>
                            )}

                            {(session || imgUrl) && (
                                <>
                                    {isPDF && (
                                            <div className={`w-full h-full ${loading ? 'opacity-0 absolute' : ''}`}>
                                                <PdfDocumentViewer
                                                    previewUrl={previewUrl}
                                                    sizeBytes={file.file_size_bytes ?? file.size_bytes}
                                                    onLoad={handlePreviewLoad}
                                                    onError={handlePreviewError}
                                                />
                                            </div>
                                        )}
                                    {isImage && <img
                                        src={imgUrl || previewUrl}
                                        className={`max-w-full max-h-full object-contain shadow-2xl transition-opacity duration-500 ${loading ? 'opacity-0 absolute' : 'opacity-100'}`}
                                        onLoad={() => setLoading(false)}
                                        onError={handlePreviewError}
                                        alt="Preview"
                                    />}
                                    {isVideo && <video
                                        controls
                                        controlsList="nodownload"
                                        className={`max-w-full max-h-full transition-opacity duration-500 ${loading ? 'opacity-0 absolute' : 'opacity-100'}`}
                                        onLoadedData={() => setLoading(false)}
                                        src={previewUrl}
                                    >Your browser does not support the video tag.</video>}
                                    {isText && (
                                        <div className={`w-full h-full p-8 font-mono text-sm leading-relaxed text-gray-300 overflow-auto whitespace-pre-wrap selection:bg-indigo-600/40 transition-opacity duration-500 ${loading ? 'opacity-0 absolute' : 'opacity-100'}`}>
                                            <WebViewer textUrl={previewUrl} onLoad={() => setLoading(false)} />
                                        </div>
                                    )}
                                    {!isPDF && !isImage && !isText && !isVideo && (
                                        <div className="text-center p-10 glass-card mx-8 border-white/5 bg-white/5">
                                            <div className="w-16 h-16 bg-gray-800 rounded-2xl flex items-center justify-center text-gray-400 mx-auto mb-4">
                                                <Icons.File size={32} />
                                            </div>
                                            <h3 className="text-lg font-bold mb-2 text-white">No Preview Available</h3>
                                            <p className="text-gray-400 text-sm mb-6">This file type cannot be previewed directly in the browser. You can still download it securely.</p>
                                            <button onClick={() => api.downloadFile(fileId, fileName)} className="w-full bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-100 border border-indigo-500/20 py-3 rounded-xl flex items-center justify-center gap-2 font-bold transition-all shadow-[0_0_15px_rgba(99,102,241,0.1)]">
                                                <Icons.Download size={20} /> Download File
                                            </button>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>
            );
        }


        // ── LOGIN ────────────────────────────────────────────────────────────────
        function Login({ onLogin, onLogout, embedded = false, initialUsername = '', lockUsername = false, oauthExchangeError = null }) {
            const [mode, setMode] = useState('login');
            const [step, setStep] = useState('username'); // 'username' | 'password' — Sign In only
            const [username, setUsername] = useState(initialUsername);
            const [email, setEmail] = useState('');
            const [password, setPassword] = useState('');
            const [confirmPassword, setConfirmPassword] = useState('');
            const [error, setError] = useState('');
            const [loading, setLoading] = useState(false);
            const [authCapabilities, setAuthCapabilities] = useState({
                registration_enabled: false,
                google_oauth_enabled: false,
            });

            useEffect(() => {
                let cancelled = false;
                fetch('/api/auth/capabilities')
                    .then(response => response.ok ? response.json() : Promise.reject(new Error('capabilities unavailable')))
                    .then(value => {
                        if (cancelled) return;
                        const next = {
                            registration_enabled: value?.registration_enabled === true,
                            google_oauth_enabled: value?.google_oauth_enabled === true,
                        };
                        setAuthCapabilities(next);
                        if (!next.registration_enabled) setMode('login');
                    })
                    .catch(() => {
                        if (!cancelled) setMode('login');
                    });
                return () => { cancelled = true; };
            }, []);

            const switchMode = (m) => { setMode(m); setStep('username'); setError(''); setPassword(''); setConfirmPassword(''); };
            const passwordsMatch = confirmPassword === '' || password === confirmPassword;

            const handleSubmit = async (e) => {
                e.preventDefault();
                setError('');
                if (mode === 'register' && password !== confirmPassword) {
                    setError('Passwords do not match');
                    return;
                }
                setLoading(true);

                try {
                    let response;
                    if (mode === 'login') {
                        response = await api.login(username, password);
                    } else {
                        response = await api.register(username, email, password);
                        if (response.user_id) {
                            response = await api.login(username, password);
                        }
                    }

                    if (response.access_token) {
                        storeAuthSession(response, username);
                        onLogin(localStorage.getItem('username'));
                    } else {
                        const errorMsg = formatApiError(response.detail || response.error, 'Authentication failed');
                        setError(errorMsg);
                    }
                } catch (err) {
                    console.error("Login error:", err);
                    setError(err.message || 'Connection failed. Is the server running?');
                } finally {
                    setLoading(false);
                }
            };

            if (embedded) {
                return (
                    <div className="w-full animate-fade-in">
                        <div className="glass-card glass-card-static p-0 border-0 bg-transparent shadow-none">
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div>
                                    <label className="block text-sm text-gray-400 mb-2">Username</label>
                                    <input
                                        type="text"
                                        value={username}
                                        onChange={e => !lockUsername && setUsername(e.target.value)}
                                        className={`input-field ${lockUsername ? 'opacity-50 cursor-not-allowed text-zinc-500' : ''}`}
                                        placeholder="Enter username"
                                        readOnly={lockUsername}
                                        required
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-2">Password</label>
                                    <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="input-field" placeholder="Enter password" required />
                                </div>
                                {error && <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">{error}</div>}
                                <div className="flex gap-3">
                                    <button type="submit" disabled={loading} className="flex-1 btn-primary font-bold py-3 rounded-xl flex items-center justify-center gap-2 disabled:opacity-50">
                                        {loading && <Icons.Loader />}
                                        Resume Session
                                    </button>

                                    {onLogout && (
                                        <button
                                            type="button"
                                            onClick={onLogout}
                                            className="flex-1 py-3 rounded-xl border border-white/10 text-zinc-400 hover:text-white hover:bg-white/5 text-[10px] font-bold uppercase tracking-widest transition-colors flex items-center justify-center"
                                        >
                                            Logout from Account
                                        </button>
                                    )}
                                </div>
                            </form>
                        </div>
                    </div>
                );
            }

            return (
                <div data-testid="auth-page" className="min-h-screen flex items-center justify-center bg-black relative overflow-hidden">

                    <div className="relative w-full max-w-md px-8 py-10 animate-fade-in">

                        {/* Logo */}
                        <div className="flex flex-col items-center mb-8">
                            <div className="relative mb-5">
                                <img src="/svg/lavix.svg" alt="Lavix Vault" className="w-44 h-44" />
                            </div>
                            <h1 className="text-2xl font-black tracking-tight mb-1">
                                <span className="gradient-text">LAVIX VAULT</span>
                            </h1>
                            <p className="text-[10px] font-bold tracking-[0.15em] uppercase text-zinc-600">
                                Where <span className="text-zinc-300">Encrypted Storage</span> Meets <span className="gradient-text font-black">AI-Powered Privacy</span>
                            </p>
                        </div>

                        {/* Card */}
                        <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] backdrop-blur-md p-7 shadow-2xl">

                            {/* Login / Register tabs */}
                            {authCapabilities.registration_enabled && (
                                <div className="flex bg-zinc-900 rounded-xl p-1 mb-6">
                                    <button data-testid="auth-sign-in-tab" onClick={() => switchMode('login')}
                                        className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${mode === 'login' ? 'bg-zinc-700 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}>
                                        Sign In
                                    </button>
                                    <button data-testid="auth-register-tab" onClick={() => switchMode('register')}
                                        className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${mode === 'register' ? 'bg-zinc-700 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}>
                                        Register
                                    </button>
                                </div>
                            )}

                            {/* Form */}
                            <form onSubmit={handleSubmit} className="space-y-4">
                                {mode === 'login' && step === 'username' ? (
                                    <>
                                        <div>
                                            <label className="block text-xs font-medium text-zinc-400 mb-1.5 uppercase tracking-wider">Username or Email</label>
                                            <input
                                                data-testid="auth-login-username"
                                                type="text" value={username} onChange={e => setUsername(e.target.value)}
                                                onKeyDown={e => e.key === 'Enter' && username.trim() && (e.preventDefault(), setStep('password'))}
                                                className="w-full bg-zinc-900 border border-zinc-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 outline-none transition-all"
                                                placeholder="username or email@example.com" autoFocus
                                            />
                                        </div>
                                        <button data-testid="auth-login-next" type="button" disabled={!username.trim()}
                                            onClick={() => setStep('password')}
                                            className="w-full bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed text-indigo-200 hover:text-white font-bold py-3 rounded-xl text-sm transition-all duration-200 flex items-center justify-center gap-2 mt-1">
                                            Next →
                                        </button>
                                    </>
                                ) : mode === 'login' && step === 'password' ? (
                                    <>
                                        <div className="flex items-center justify-between bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5">
                                            <span className="text-sm text-zinc-200 font-medium">{username}</span>
                                            <button type="button" onClick={() => { setStep('username'); setError(''); setPassword(''); }}
                                                className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">
                                                Change
                                            </button>
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-zinc-400 mb-1.5 uppercase tracking-wider">Password</label>
                                            <input
                                                data-testid="auth-login-password"
                                                type="password" value={password} onChange={e => setPassword(e.target.value)}
                                                className="w-full bg-zinc-900 border border-zinc-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 outline-none transition-all"
                                                placeholder="••••••••" autoFocus required
                                            />
                                        </div>
                                        {error && (
                                            <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 text-red-400 px-3 py-2.5 rounded-xl text-xs">
                                                <svg className="mt-0.5 shrink-0" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                                                {error}
                                            </div>
                                        )}
                                        <button data-testid="auth-login-submit" type="submit" disabled={loading}
                                            className="w-full bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed text-indigo-200 hover:text-white font-bold py-3 rounded-xl text-sm transition-all duration-200 flex items-center justify-center gap-2 mt-1">
                                            {loading ? <><Icons.Loader /><span>Authenticating…</span></> : <span>Enter Vault</span>}
                                        </button>
                                    </>
                                ) : (
                                    /* Register mode — all fields visible */
                                    <>
                                        <div>
                                            <label className="block text-xs font-medium text-zinc-400 mb-1.5 uppercase tracking-wider">Username</label>
                                            <input
                                                data-testid="auth-register-username"
                                                type="text" value={username} onChange={e => setUsername(e.target.value)}
                                                className="w-full bg-zinc-900 border border-zinc-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 outline-none transition-all"
                                                placeholder="your-username" required
                                            />
                                        </div>
                                        <div className="animate-fade-in">
                                            <label className="block text-xs font-medium text-zinc-400 mb-1.5 uppercase tracking-wider">Email</label>
                                            <input
                                                data-testid="auth-register-email"
                                                type="email" value={email} onChange={e => setEmail(e.target.value)}
                                                className="w-full bg-zinc-900 border border-zinc-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 outline-none transition-all"
                                                placeholder="you@example.com" required
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-zinc-400 mb-1.5 uppercase tracking-wider">Password</label>
                                            <input
                                                data-testid="auth-register-password"
                                                type="password" value={password} onChange={e => setPassword(e.target.value)}
                                                className="w-full bg-zinc-900 border border-zinc-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 outline-none transition-all"
                                                placeholder="••••••••" required
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-zinc-400 mb-1.5 uppercase tracking-wider">Confirm Password</label>
                                            <input
                                                data-testid="auth-register-confirm-password"
                                                type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)}
                                                className={`w-full bg-zinc-900 border rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 outline-none transition-all focus:ring-1 ${
                                                    confirmPassword === '' ? 'border-zinc-700 focus:border-indigo-500 focus:ring-indigo-500/30'
                                                    : passwordsMatch ? 'border-green-600 focus:border-green-500 focus:ring-green-500/20'
                                                    : 'border-red-500/60 focus:border-red-500 focus:ring-red-500/20'
                                                }`}
                                                placeholder="••••••••" required
                                            />
                                            {confirmPassword !== '' && (
                                                <p className={`text-[10px] mt-1.5 font-medium ${passwordsMatch ? 'text-green-400' : 'text-red-400'}`}>
                                                    {passwordsMatch ? '✓ Passwords match' : '✗ Passwords do not match'}
                                                </p>
                                            )}
                                        </div>
                                        {error && (
                                            <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 text-red-400 px-3 py-2.5 rounded-xl text-xs">
                                                <svg className="mt-0.5 shrink-0" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                                                {error}
                                            </div>
                                        )}
                                        <button data-testid="auth-register-submit" type="submit" disabled={loading || !passwordsMatch || !confirmPassword}
                                            className="w-full bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed text-indigo-200 hover:text-white font-bold py-3 rounded-xl text-sm transition-all duration-200 flex items-center justify-center gap-2 mt-1">
                                            {loading ? <><Icons.Loader /><span>Authenticating…</span></> : <span>Create Account</span>}
                                        </button>
                                    </>
                                )}
                            </form>

                            {/* SSO divider + Google button */}
                            {oauthExchangeError && (
                                <div className="mt-4 bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-2.5 rounded-xl text-xs leading-relaxed">
                                    {oauthExchangeError}
                                </div>
                            )}
                            {authCapabilities.google_oauth_enabled && (
                                <>
                                    <div className="flex items-center gap-3 my-5">
                                        <div className="flex-1 h-px bg-zinc-800" />
                                        <span className="text-[11px] text-zinc-600 font-medium">or continue with</span>
                                        <div className="flex-1 h-px bg-zinc-800" />
                                    </div>
                                    <div className="grid gap-3 grid-cols-1">
                                        <button onClick={() => { window.location.href = '/api/auth/google/login'; }}
                                            className="flex items-center justify-center gap-2 bg-zinc-900 hover:bg-zinc-800 active:scale-[0.97] border border-zinc-700 hover:border-zinc-600 text-white text-sm font-medium py-2.5 rounded-xl transition-all">
                                            <svg width="16" height="16" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.6 20H24v8h11.3C33.6 33.1 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3 0 5.8 1.1 7.9 3l5.7-5.7C34.1 6.5 29.3 4.5 24 4.5 13.2 4.5 4.5 13.2 4.5 24S13.2 43.5 24 43.5c10.5 0 19-7.6 19.5-18.2.1-.4.1-.9.1-1.3 0-1.4-.1-2.7-.5-4z"/><path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.5 15.1 18.9 12 24 12c3 0 5.8 1.1 7.9 3l5.7-5.7C34.1 6.5 29.3 4.5 24 4.5c-7.6 0-14.2 4.3-17.7 10.2z"/><path fill="#4CAF50" d="M24 43.5c5.2 0 9.9-1.9 13.5-5l-6.2-5.3C29.4 35 26.8 36 24 36c-5.3 0-9.7-3-11.4-7.3l-6.5 5C9.9 39.3 16.5 43.5 24 43.5z"/><path fill="#1976D2" d="M43.6 20H24v8h11.3c-.9 2.5-2.5 4.6-4.6 6l6.2 5.3C40.8 36.2 44 30.6 44 24c0-1.4-.1-2.7-.4-4z"/></svg>
                                            Google
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>

                        <p className="text-center text-zinc-600 text-[11px] mt-6 tracking-wide">
                            🔐 Zero-Knowledge &nbsp;·&nbsp; Hybrid Encryption
                        </p>
                    </div>
                </div>
            );
        }


        // ── SIDEBAR ──────────────────────────────────────────────────────────────
        function Sidebar({ view, setView, user, userProfile, onLogout, sessions, currentSessionId, onSelectSession, onNewChat, onDeleteSession, isCollapsed, setIsCollapsed, isMobileOpen, setIsMobileOpen, storageUsed, storageQuota, aiStats, version, pgChats = [], onDeletePgChat, onPinChat, onOpenPgChat, currentPgChatId, pinnedSessions = [], onTogglePin, currentTab, setCurrentTab, onRefreshHome }) {
            const navItems = [
                { id: 'files', icon: Icons.Home, label: 'Dashboard' },
                { id: 'upload', icon: Icons.Upload, label: 'Upload' },
                { id: 'chat', icon: Icons.Chat, label: 'AI Chat' },
                { id: 'settings', icon: Icons.Settings, label: 'Settings' },
            ];
            // Logo image acts as Home (YouTube-style): Dashboard root + fresh list.
            const goHome = () => {
                setView('files');
                setCurrentTab('files');
                if (onRefreshHome) onRefreshHome();
                window.dispatchEvent(new CustomEvent('lavix-go-home'));
                if (window.innerWidth < 769) setIsMobileOpen(false);
            };

            return (
                <>
                    {/* Mobile Overlay */}
                    {isMobileOpen && (
                        <div
                            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[190] md:hidden animate-fade-in"
                            onClick={() => setIsMobileOpen(false)}
                        ></div>
                    )}

                    <div className={`sidebar-transition h-full sidebar-bg border-r border-white/5 flex flex-col z-[200] ${isMobileOpen ? 'sidebar-drawer open w-[280px]' : 'sidebar-drawer w-[280px]'} ${!isMobileOpen ? (isCollapsed ? 'md:w-[80px]' : 'md:w-[280px]') : ''}`}>
                        {/* Sidebar Header (Dual Identity) */}
                        <div className={`flex items-center branding-area px-5 py-6 sticky top-0 z-20 ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
                            <div className={`flex items-center transition-all duration-500 ${isCollapsed ? 'justify-center scale-110' : ''}`}>
                                {isCollapsed ? (
                                    <div
                                        role="button"
                                        tabIndex={0}
                                        title="Home"
                                        onClick={goHome}
                                        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') goHome(); }}
                                        className="flex-shrink-0 flex items-center justify-center relative w-10 h-10 bg-black rounded-xl group cursor-pointer"
                                    >
                                        <img src="/svg/lavix.svg" alt="Lavix home" className="w-full h-full object-contain filter brightness-110" />
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-5 animate-fade-in">
                                        {/* Product - Icon + Text (logo image only is Home) */}
                                        <div className="flex items-center gap-2">
                                            <div
                                                role="button"
                                                tabIndex={0}
                                                title="Home"
                                                onClick={goHome}
                                                onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') goHome(); }}
                                                className="w-16 h-16 relative flex-shrink-0 cursor-pointer"
                                            >
                                                <img src="/svg/lavix.svg" alt="Lavix home" className="w-full h-full object-contain" />
                                            </div>
                                            <span className="text-2xl font-black tracking-tighter gradient-text mt-0.5">LAVIX VAULT</span>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <button
                                onClick={() => setIsCollapsed(!isCollapsed)}
                                className={`absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 bg-[#0a0a0c] border border-white/10 rounded-full flex items-center justify-center text-zinc-500 hover:text-red-400 hover:border-red-500/50 transition-all z-30 mobile-hide shadow-[0_0_20px_rgba(0,0,0,0.8)] ${isCollapsed ? 'rotate-180' : ''}`}
                                title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
                            >
                                <Icons.ChevronLeft size={12} />
                            </button>
                            <button
                                onClick={() => setIsMobileOpen(false)}
                                className="p-2 hover:bg-white/5 rounded-xl text-zinc-500 hover:text-white transition-all desktop-hide"
                            >
                                <Icons.X size={20} />
                            </button>
                        </div>
                        <div className="h-px bg-white/[0.06] mx-5 mb-2"></div>

                        <div className="p-4 flex-1 overflow-y-auto overflow-x-hidden pt-2">
                            <div className="space-y-1.5 mb-3">
                                {navItems.map(item => {
                                    // Guard against an undefined icon component (React #130):
                                    // fall back to a static icon instead of crashing render.
                                    const Icon = item.icon || Icons.File;
                                    const isActive = view === item.id && !(item.id === 'files' && currentTab === 'trash');
                                    return (
                                        <button
                                            key={item.id}
                                            data-testid={`nav-${item.id}`}
                                            onClick={() => {
                                                if (item.id === 'chat') onNewChat();
                                                else {
                                                    setView(item.id);
                                                    if (item.id === 'files') setCurrentTab('files');
                                                }
                                                if (window.innerWidth < 769) setIsMobileOpen(false);
                                            }}
                                            className={`w-full flex items-center rounded-xl transition-all group overflow-hidden ${isCollapsed ? 'justify-center p-3' : 'px-4 py-3'} ${isActive ? 'bg-indigo-500/20 border border-indigo-400/50 text-indigo-200' : 'text-zinc-500 hover:text-white hover:bg-white/5'}`}
                                            title={isCollapsed ? item.label : ''}
                                        >
                                            <div className={`${isActive ? 'text-indigo-400' : 'group-hover:text-white transition-colors'}`}>
                                                <Icon size={20} />
                                            </div>
                                            {!isCollapsed && (
                                                <span className="font-medium ml-3 animate-fade-in whitespace-nowrap">{item.label}</span>
                                            )}
                                        </button>
                                    );
                                })}

                                {/* Separator + Trash nav item */}
                                {(() => {
                                    const isTrashActive = view === 'files' && currentTab === 'trash';
                                    return (
                                        <button
                                            onClick={() => {
                                                setView('files');
                                                setCurrentTab('trash');
                                                if (window.innerWidth < 769) setIsMobileOpen(false);
                                            }}
                                            className={`w-full flex items-center rounded-xl transition-all group overflow-hidden ${isCollapsed ? 'justify-center p-3' : 'px-4 py-3'} ${isTrashActive ? 'bg-violet-500/10 text-white border border-violet-500/20' : 'text-zinc-500 hover:text-white hover:bg-white/5'}`}
                                            title={isCollapsed ? 'Trash' : ''}
                                        >
                                            <div className={`${isTrashActive ? 'text-violet-400' : 'group-hover:text-violet-400 transition-colors'}`}>
                                                <Icons.Trash size={20} />
                                            </div>
                                            {!isCollapsed && (
                                                <span className="font-medium ml-3 animate-fade-in whitespace-nowrap">Trash</span>
                                            )}
                                        </button>
                                    );
                                })()}
                            </div>

                            {view === 'chat' && (
                                <div className="animate-fade-in border-t border-white/5 pt-3 mt-3">
                                    <div className={`flex items-center mb-4 px-2 ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
                                        {!isCollapsed && (
                                            <span className="text-[10px] font-bold text-zinc-500 tracking-[0.2em] uppercase">Chat History</span>
                                        )}
                                        <button onClick={onNewChat} className="p-1.5 hover:bg-violet-500/20 text-violet-400 rounded-lg transition-colors" title="New Chat">
                                            <Icons.X size={16} className="rotate-45" />
                                        </button>
                                    </div>
                                    {/* PostgreSQL persistent chats — Favourites + Recent */}
                                    {pgChats.length > 0 && !isCollapsed && (() => {
                                        const uniquePgChats = pgChats.filter((c, i, arr) => arr.findIndex(x => x.id === c.id) === i);
                                        const pinnedChats = uniquePgChats.filter(c => c.pinned);
                                        const recentChats = uniquePgChats.filter(c => !c.pinned);
                                        const renderPgChat = (chat) => (
                                            <div key={chat.id} className="group relative">
                                                <button
                                                    data-testid={`chat-history-${chat.id}`}
                                                    onClick={() => { onOpenPgChat && onOpenPgChat(chat.id); if (window.innerWidth < 769) setIsMobileOpen(false); }}
                                                    className={`w-full text-left px-3 py-2 rounded-xl text-xs truncate block transition-all pr-14 ${currentPgChatId === chat.id ? 'bg-white/10 text-white font-medium' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'}`}
                                                >
                                                    {chat.pinned && <span className="text-amber-400 mr-1">★</span>}{chat.title || 'New Chat'}
                                                </button>
                                                <div className="absolute right-1 top-1.5 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-all">
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); onPinChat && onPinChat(chat.id, chat.pinned); }}
                                                        className={`p-1.5 rounded-lg transition-all hover:bg-white/10 ${chat.pinned ? 'text-amber-400' : 'text-zinc-600 hover:text-amber-400'}`}
                                                        title={chat.pinned ? 'Unpin' : 'Pin'}
                                                    >
                                                        <Icons.Pin size={10} />
                                                    </button>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); onDeletePgChat && onDeletePgChat(chat.id); }}
                                                        className="p-1.5 text-zinc-600 hover:text-red-400 rounded-lg hover:bg-red-500/10 transition-all"
                                                        title="Delete chat"
                                                    >
                                                        <Icons.Trash size={11} />
                                                    </button>
                                                </div>
                                            </div>
                                        );
                                        return (
                                            <div className="space-y-1 mb-3">
                                                {pinnedChats.length > 0 && (
                                                    <>
                                                        <div className="px-3 py-1 text-[10px] font-semibold text-amber-400 uppercase tracking-wider">Favourites</div>
                                                        {pinnedChats.map(renderPgChat)}
                                                    </>
                                                )}
                                                {pinnedChats.length > 0 && recentChats.length > 0 && (
                                                    <hr className="border-zinc-800 my-1" />
                                                )}
                                                {recentChats.length > 0 && (
                                                    <>
                                                        <div className="px-3 py-1 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Recent</div>
                                                        {recentChats.slice(0, 50).map(renderPgChat)}
                                                    </>
                                                )}
                                            </div>
                                        );
                                    })()}
                                    {/* Redis sessions — hidden; PG chats are the sole persistent store */}
                                    {false && (() => {
                                        const smartTitle = (t) => {
                                            if (!t || t === 'New Chat') return 'New Chat';
                                            // Strip leading @filename prefixes (e.g. "@Resume.pdf ")
                                            let s = t.replace(/^(@\S+\s*)+/, '').trim() || t.trim();
                                            // Capitalise first letter
                                            s = s.charAt(0).toUpperCase() + s.slice(1);
                                            // Detect old 30/40-char mid-word truncation: ends non-whitespace
                                            // and length is >= 28 (near the old cutoff) — add ellipsis
                                            if (s.length >= 28 && /\w$/.test(s) && !s.endsWith('?') && !s.endsWith('.') && !s.endsWith('!')) {
                                                s = s + '…';
                                            }
                                            return s;
                                        };
                                        // Hide Redis shadow-sessions created when opening PG chats (user_X_pgchat_UUID)
                                        const visibleSessions = sessions.filter(s => !s.id.includes('_pgchat_'));
                                        const pinned = visibleSessions.filter(s => pinnedSessions.includes(s.id));
                                        const recent = visibleSessions.filter(s => !pinnedSessions.includes(s.id));
                                        const SessionRow = ({ session }) => {
                                            const isPinned = pinnedSessions.includes(session.id);
                                            return (
                                                <div key={session.id} className="group relative">
                                                    <button
                                                        onClick={() => {
                                                            onSelectSession(session.id);
                                                            if (window.innerWidth < 769) setIsMobileOpen(false);
                                                        }}
                                                        className={`w-full text-left px-3 py-2 rounded-xl text-xs truncate block transition-all ${currentSessionId === session.id ? 'bg-white/10 text-white font-medium' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'} ${isCollapsed ? 'hidden' : 'pr-14'}`}
                                                    >
                                                        {smartTitle(session.title)}
                                                    </button>
                                                    {!isCollapsed && (
                                                        <div className="absolute right-1 top-1 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-all">
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); onTogglePin && onTogglePin(session.id); }}
                                                                className={`p-1.5 rounded-lg transition-all hover:bg-white/10 ${isPinned ? 'text-amber-400' : 'text-zinc-600 hover:text-amber-400'}`}
                                                                title={isPinned ? 'Unpin' : 'Pin to Favourites'}
                                                            >
                                                                <svg width="10" height="10" viewBox="0 0 24 24" fill={isPinned ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                                                            </button>
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); onDeleteSession(session.id); }}
                                                                className="p-1.5 text-zinc-600 hover:text-red-400 rounded-lg hover:bg-red-500/10 transition-all"
                                                                title="Delete"
                                                            >
                                                                <Icons.Trash size={10} />
                                                            </button>
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        };
                                        return (
                                            <>
                                                {pinned.length > 0 && !isCollapsed && (
                                                    <div className="mb-3">
                                                        <div className="flex items-center gap-1.5 px-3 mb-1.5">
                                                            <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor" className="text-amber-500"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                                                            <span className="text-[9px] font-bold text-amber-500/70 tracking-[0.18em] uppercase">Favourites</span>
                                                        </div>
                                                        <div className="space-y-0.5">
                                                            {pinned.map(s => <SessionRow key={s.id} session={s} />)}
                                                        </div>
                                                    </div>
                                                )}
                                                {recent.length > 0 && !isCollapsed && (
                                                    <div>
                                                        {pinned.length > 0 && (
                                                            <div className="px-3 mb-1.5">
                                                                <span className="text-[9px] font-bold text-zinc-600 tracking-[0.18em] uppercase">Recent</span>
                                                            </div>
                                                        )}
                                                        <div className="space-y-0.5">
                                                            {recent.map(s => <SessionRow key={s.id} session={s} />)}
                                                        </div>
                                                    </div>
                                                )}
                                            </>
                                        );
                                    })()}
                                </div>
                            )}
                        </div>

                        {!isCollapsed && (
                            <div className="px-6 mb-2">
                                {/* Storage */}
                                <div className="rounded-xl border border-white/[0.06] px-3 py-2.5">
                                    <div className="flex justify-between items-center mb-2 px-0.5">
                                        <span className="text-[11px] font-bold text-zinc-500 tracking-[0.2em] uppercase">Storage</span>
                                        <span className="text-[11px] font-bold text-zinc-300 tracking-wider uppercase">
                                            {((storageUsed / (storageQuota || 1)) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden border border-white/5 px-[1px] py-[1px]">
                                        <div
                                            className="h-full bg-gradient-to-r from-violet-600 to-indigo-500 rounded-full transition-all duration-1000 shadow-[0_0_10px_rgba(124,58,237,0.3)]"
                                            style={{ width: `${Math.min(100, (storageUsed / (storageQuota || 1)) * 100)}%` }}
                                        ></div>
                                    </div>
                                    <div className="flex justify-between mt-2 px-0.5">
                                        <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-tight">
                                            {formatSize(storageUsed)} USED
                                        </span>
                                        <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-tight">
                                            {formatSize(storageQuota)}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {!isCollapsed && (
                            <div className="p-4 border-t border-white/5 bg-black/40">
                                <div className="flex items-center gap-3 px-2 py-2">
                                    {userProfile?.avatar_data
                                        ? <img src={userProfile.avatar_data} className="w-8 h-8 rounded-full object-cover flex-shrink-0" alt="avatar" />
                                        : <div className="w-8 h-8 rounded-full bg-violet-600/20 flex items-center justify-center text-violet-400 font-bold text-xs flex-shrink-0">{user?.charAt(0).toUpperCase()}</div>
                                    }
                                    <div className="flex-1 overflow-hidden">
                                        <p className="text-[11px] font-bold text-white line-clamp-2 break-all">{user}</p>
                                        <div className="flex items-center gap-2">
                                            <p className="text-[9px] text-zinc-500 uppercase tracking-wider">Vault Operator</p>
                                            {version && <span className="text-[8px] bg-white/5 px-1 rounded text-zinc-600 font-mono">v{version}</span>}
                                        </div>
                                    </div>
                                    <button onClick={onLogout} className="p-2 text-zinc-600 hover:text-violet-400 hover:bg-violet-500/10 rounded-lg transition-all">
                                        <Icons.LogOut size={14} />
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </>
            );
        }



        // Helper to format file size
        const formatSize = (bytes) => {
            if (!bytes || bytes === 0) return '0 B';
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
            return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
        };

        const formatRelativeTime = (dateStr) => {
            if (!dateStr) return '—';
            const diff = Date.now() - new Date(dateStr).getTime();
            const mins = Math.floor(diff / 60000);
            const hrs = Math.floor(diff / 3600000);
            const days = Math.floor(diff / 86400000);
            if (mins < 1) return 'just now';
            if (mins < 60) return `${mins}m ago`;
            if (hrs < 24) return `${hrs}h ago`;
            if (days < 7) return `${days}d ago`;
            return new Date(dateStr).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
        };

        // ── FILE CATEGORY HELPERS ────────────────────────────────────────────
        const getFileCategory = (file) => {
            const m = (file.mime_type || '').toLowerCase();
            const n = (file.filename || '').toLowerCase();
            if (m.includes('pdf') || n.endsWith('.pdf')) return 'PDF';
            if (m.startsWith('image/')) return 'Images';
            if (m.startsWith('video/')) return 'Videos';
            if (m.startsWith('audio/')) return 'Audio';
            if (m.includes('sheet') || m.includes('excel') || m === 'text/csv' || /\.(xls|xlsx|csv)$/.test(n)) return 'Spreadsheets';
            if (m.includes('wordprocess') || /\.(doc|docx)$/.test(n)) return 'Documents';
            if (m.startsWith('text/') || /\.(py|js|ts|jsx|tsx|json|md|txt|sh|yaml|yml|toml|go|rs|java|cpp|c|h)$/.test(n)) return 'Code & Text';
            if (m.includes('presentation') || /\.(ppt|pptx|key)$/.test(n)) return 'Presentations';
            return 'Other';
        };
        const FILE_CATEGORIES = [
            { key: 'PDF',           dot: 'bg-red-500'    },
            { key: 'Images',        dot: 'bg-pink-500'   },
            { key: 'Videos',        dot: 'bg-purple-500' },
            { key: 'Audio',         dot: 'bg-yellow-500' },
            { key: 'Spreadsheets',  dot: 'bg-green-500'  },
            { key: 'Documents',     dot: 'bg-blue-500'   },
            { key: 'Code & Text',   dot: 'bg-zinc-500'   },
            { key: 'Presentations', dot: 'bg-orange-500' },
            { key: 'Other',         dot: 'bg-zinc-600'   },
        ];

        // ── FALLBACK ICON (fills its container, used when no thumbnail is available)
        function FallbackIcon({ mimeType, filename }) {
            const ext = ((filename || '').split('.').pop() || 'file').toUpperCase().slice(0, 4);
            const m = (mimeType || '').toLowerCase();
            const n = (filename || '').toLowerCase();
            let bg = 'bg-indigo-500/20', txt = 'text-indigo-400';
            if (m.startsWith('image/'))                                  { bg = 'bg-pink-500/20';   txt = 'text-pink-400';   }
            else if (m.includes('pdf') || n.endsWith('.pdf'))            { bg = 'bg-red-500/20';    txt = 'text-red-400';    }
            else if (m.includes('sheet') || m.includes('excel') ||
                     m === 'text/csv' || /\.(xls|xlsx|csv)$/.test(n))   { bg = 'bg-green-500/20';  txt = 'text-green-400';  }
            else if (m.includes('wordprocess') || n.endsWith('.docx'))  { bg = 'bg-blue-500/20';   txt = 'text-blue-400';   }
            else if (m.startsWith('text/') || /\.(txt|md|json|js|py)$/.test(n)) { bg = 'bg-gray-500/20'; txt = 'text-gray-300'; }
            else if (m.startsWith('video/'))                             { bg = 'bg-purple-500/20'; txt = 'text-purple-400'; }
            else if (m.startsWith('audio/'))                             { bg = 'bg-yellow-500/20'; txt = 'text-yellow-400'; }
            return (
                <div className={`w-full h-full ${bg} flex flex-col items-center justify-center gap-1`}>
                    <svg className={`w-5 h-5 ${txt} opacity-60`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span className={`text-[9px] font-bold tracking-wide ${txt}`}>{ext}</span>
                </div>
            );
        }

        // ── FILE THUMBNAIL ────────────────────────────────────────────────────
        // Uses IntersectionObserver so thumbnails are generated lazily as rows
        // scroll into view, and caches results for the session.
        //
        // size="sm"  → 40×40 px  (list view rows)
        // size="lg"  → full-width × 112 px  (grid view cards)
        function FileThumbnail({ file, size, onStateChange }) {
            const [state, setState] = useState('idle'); // idle | loading | decoding | done | error
            const [src,   setSrc  ] = useState(null);
            const [retry, setRetry] = useState(0);
            const ref = useRef(null);
            const observerRef = useRef(null);
            const renderAttemptRef = useRef(0);
            const visibleGenerationRef = useRef(0);

            const fileId = Number(file?.file_id ?? file?.id);
            const supportsThumbnail = mvCanGenerateThumbnail(file);

            const dims = size === 'lg' ? 'w-full h-28' : size === 'panel' ? 'w-full h-52' : 'w-10 h-10 flex-shrink-0';
            const fit  = size === 'panel' ? 'object-contain object-top' : 'object-cover';

            useEffect(() => {
                renderAttemptRef.current += 1;
                visibleGenerationRef.current = 0;
                observerRef.current?.disconnect();
                observerRef.current = null;
                setSrc(null);
                setState('idle');
                setRetry(0);
            }, [fileId, supportsThumbnail]);

            useEffect(() => {
                if (typeof onStateChange === 'function') onStateChange(state);
            }, [state, onStateChange]);

            useEffect(() => {
                const handleInvalidation = (event) => {
                    if (Number(event?.detail?.fileId) !== fileId) return;
                    if (!supportsThumbnail) return;
                    visibleGenerationRef.current = Math.max(
                        visibleGenerationRef.current,
                        Number(event?.detail?.generation) || 0,
                    );
                    renderAttemptRef.current += 1;
                    observerRef.current?.disconnect();
                    observerRef.current = null;
                    setSrc(null);
                    setState(event?.detail?.failed ? 'error' : 'idle');
                    if (event?.detail?.reload !== false) setRetry(value => value + 1);
                };
                const handleGenerated = (event) => {
                    if (Number(event?.detail?.fileId) !== fileId) return;
                    if (!supportsThumbnail) return;
                    const generation = Number(event?.detail?.generation) || 0;
                    const thumbnail = event?.detail?.thumbnail;
                    if (generation < visibleGenerationRef.current || typeof thumbnail !== 'string' || !thumbnail) return;
                    visibleGenerationRef.current = generation;
                    renderAttemptRef.current += 1;
                    observerRef.current?.disconnect();
                    observerRef.current = null;
                    setSrc(thumbnail);
                    setState('decoding');
                };
                window.addEventListener('lavix-thumbnail-invalidated', handleInvalidation);
                window.addEventListener('lavix-thumbnail-generated', handleGenerated);
                return () => {
                    window.removeEventListener('lavix-thumbnail-invalidated', handleInvalidation);
                    window.removeEventListener('lavix-thumbnail-generated', handleGenerated);
                };
            }, [fileId, supportsThumbnail]);

            useEffect(() => {
                const el = ref.current;
                if (!el || !supportsThumbnail || !Number.isInteger(fileId) || fileId <= 0 || file?.is_deleted) return;

                // Return cached result immediately
                const cached = window._mvThumbCache?.get(fileId);
                if (cached) { setSrc(cached); setState('decoding'); return; }

                let cancelled = false;

                const obs = new IntersectionObserver(async (entries) => {
                    if (!entries[0].isIntersecting) return;
                    obs.disconnect();
                    observerRef.current = null;
                    if (cancelled) return;

                    const renderAttempt = renderAttemptRef.current;
                    setState('loading');
                    let generationDeadlineTimer;
                    try {
                        const generation = window.mvGenerateFileThumbnail(file, api.createPreviewSession);
                        const deadline = new Promise((_, reject) => {
                            generationDeadlineTimer = window.setTimeout(() => {
                                const error = new Error('thumbnail generation timed out');
                                error.code = 'LAVIX_THUMBNAIL_TIMEOUT';
                                reject(error);
                            }, 45_000);
                        });
                        const thumb = await Promise.race([generation, deadline]);
                        if (cancelled || renderAttemptRef.current !== renderAttempt) return;

                        setSrc(thumb);
                        setState('decoding');
                    } catch (error) {
                        if (cancelled || renderAttemptRef.current !== renderAttempt) return;
                        if (error?.code === 'LAVIX_THUMBNAIL_TIMEOUT' && typeof window.invalidateMvFileThumbnail === 'function') {
                            window.invalidateMvFileThumbnail(fileId, {
                                reload: false,
                                reason: 'automatic-generation-timeout',
                                failed: true,
                                detachPendingSession: true,
                            });
                        } else {
                            setState('error');
                        }
                    } finally {
                        if (generationDeadlineTimer) window.clearTimeout(generationDeadlineTimer);
                    }
                }, { rootMargin: '120px', threshold: 0.01 });

                observerRef.current = obs;
                obs.observe(el);
                return () => {
                    cancelled = true;
                    obs.disconnect();
                    if (observerRef.current === obs) observerRef.current = null;
                };
            }, [fileId, file?.mime_type, file?.filename, file?.size_bytes, file?.file_size_bytes, file?.is_deleted, retry, supportsThumbnail]);

            return (
                <div
                    ref={ref}
                    data-testid={`file-thumbnail-${fileId}`}
                    data-thumbnail-state={state}
                    aria-busy={state === 'loading' || state === 'decoding' ? 'true' : 'false'}
                    className={`${dims} relative rounded-lg overflow-hidden bg-[#0d0d0f] ${size === 'sm' && state === 'error' ? 'border-0' : 'border border-white/5'}`}
                >
                    {(state === 'decoding' || state === 'done') && src ? (
                        <>
                            <img
                                src={src}
                                alt=""
                                onLoad={() => setState('done')}
                                onError={() => setState('error')}
                                className={`w-full h-full ${fit} ${state === 'decoding' ? 'opacity-0' : 'opacity-100'}`}
                            />
                            {state === 'decoding' && <div className="absolute inset-0 mv-thumb-shimmer" />}
                        </>
                    ) : state === 'loading' ? (
                        <div className="w-full h-full mv-thumb-shimmer" />
                    ) : (
                        <FallbackIcon mimeType={file.mime_type} filename={file.filename} />
                    )}
                </div>
            );
        }




        // ── DASHBOARD ────────────────────────────────────────────────────────────
        function Dashboard({ files, setFiles, refresh, onPreview, currentTab, setCurrentTab, onConfirmAction, totalSizeBytes, aiStats, onNavigateToChat, selectedItem, onSelectItem, filesLoading, onFoldersChange, searchTerm, setSearchTerm, handleAiReadyClick, isIndexing, fileSearch, onUploadIntoFolder }) {
            const [viewMode, setViewMode] = useState(localStorage.getItem('viewMode') || 'list');
            const [sortField, setSortField] = useState('uploaded_at');
            const [sortDir, setSortDir] = useState('desc');
            const [moreOpen, setMoreOpen] = useState(false);
            const moreRef = React.useRef(null);
            const docTypeScrollRef = React.useRef(null);
            const [docTypeOverflow, setDocTypeOverflow] = useState({ left: false, right: false });
            const [currentPage, setCurrentPage] = useState(1);
            const [pageSize, setPageSize] = useState(25);
            const [folders, setFolders] = useState([]);
            const [currentFolder, setCurrentFolder] = useState(null);
            const [newFolderOpen, setNewFolderOpen] = useState(false);
            const [newFolderName, setNewFolderName] = useState('');
            const [selectedItems, setSelectedItems] = useState(new Set()); // "file:123" | "folder:45"
            // Select mode: checkboxes only render while active (cleaner rows by default).
            const [selectMode, setSelectMode] = useState(false);
            const [bulkMoveFolder, setBulkMoveFolder] = useState('');  // '' = root
            const [bulkBusy, setBulkBusy] = useState(false);  // bulk AI grant/revoke in flight
            const [renamingId, setRenamingId] = useState(null); // "file:123" | "folder:45"
            const [renameValue, setRenameValue] = useState('');
            const [groupBy, setGroupBy] = useState(() => localStorage.getItem('prefGroupBy') === 'mime' ? 'mime' : 'none');
            const [docTypeFilter, setDocTypeFilter] = useState('all');
            // Indexing auto-filter: chip appears while files are actively
            // indexing so users can see WHAT is processing without hunting.
            const [indexingOnly, setIndexingOnly] = useState(false);
            useEffect(() => { localStorage.setItem('prefGroupBy', groupBy); }, [groupBy]);
            // Sync groupBy when changed from Settings
            useEffect(() => {
                const onStorage = (e) => { if (e.key === 'prefGroupBy') setGroupBy(e.newValue === 'mime' ? 'mime' : 'none'); };
                window.addEventListener('storage', onStorage);
                return () => window.removeEventListener('storage', onStorage);
            }, []);
            // Auto-switch to grid view on mobile
            useEffect(() => {
                const handleResize = () => {
                    if (window.innerWidth < 768) {
                        setViewMode('grid');
                    }
                };
                handleResize(); // Initial check
                window.addEventListener('resize', handleResize);
                return () => window.removeEventListener('resize', handleResize);
            }, []);

            useEffect(() => { setCurrentPage(1); }, [searchTerm, sortField, sortDir, currentTab, pageSize, currentFolder, docTypeFilter, indexingOnly]);
            useEffect(() => { setCurrentFolder(null); setDocTypeFilter('all'); }, [currentTab]);
            // Logo-home signal: reset to vault root with clean filters.
            useEffect(() => {
                const goRoot = () => {
                    setCurrentFolder(null);
                    setDocTypeFilter('all');
                    setIndexingOnly(false);
                    setCurrentPage(1);
                    if (setSearchTerm) setSearchTerm('');
                };
                window.addEventListener('lavix-go-home', goRoot);
                return () => window.removeEventListener('lavix-go-home', goRoot);
            }, []);
            useEffect(() => { setDocTypeFilter('all'); }, [currentFolder?.id]);

            // Load & refresh folders
            const refreshFolders = () => api.getFolders().then(f => { setFolders(f); onFoldersChange && onFoldersChange(f); }).catch(() => {});
            useEffect(() => { refreshFolders(); }, []);

            // Trashed folders
            const [trashedFolders, setTrashedFolders] = useState([]);
            const refreshTrashedFolders = () => api.getTrashedFolders().then(f => setTrashedFolders(f)).catch(() => setTrashedFolders([]));
            useEffect(() => { if (currentTab === 'trash') refreshTrashedFolders(); }, [currentTab]);

            // Derived: folder lookup map and visible children
            const folderMap = React.useMemo(() => {
                const m = new Map();
                folders.forEach(f => m.set(f.id, f));
                return m;
            }, [folders]);

            // Only folders that are direct children of currentFolder
            const visibleFolders = React.useMemo(() =>
                folders.filter(f => f.parent_id === (currentFolder ? currentFolder.id : null)),
            [folders, currentFolder]);

            // Breadcrumb path: array of folder objects from root to current
            const breadcrumb = React.useMemo(() => {
                if (!currentFolder) return [];
                const path = [];
                let f = currentFolder;
                while (f) { path.unshift(f); f = folderMap.get(f.parent_id); }
                return path;
            }, [currentFolder, folderMap]);

            // Close "⋯ More" overflow dropdown on outside click
            useEffect(() => {
                if (!moreOpen) return;
                const handler = (e) => {
                    if (moreRef.current && !moreRef.current.contains(e.target)) setMoreOpen(false);
                };
                document.addEventListener('mousedown', handler);
                return () => document.removeEventListener('mousedown', handler);
            }, [moreOpen]);

            const handleCreateFolder = async () => {
                const name = newFolderName.trim();
                if (!name) return;
                try {
                    await api.createFolder(name, currentFolder ? currentFolder.id : null);
                    setNewFolderName('');
                    setNewFolderOpen(false);
                    refreshFolders();
                } catch(e) { alert(e.message); }
            };

            const handleDeleteFolder = (folder) => {
                onConfirmAction({
                    title: `Move "${folder.name}" to Trash?`,
                    message: 'The folder and all its subfolders will be moved to Trash. Files inside will also be trashed and can be restored later.',
                    confirmText: 'Move to Trash',
                    onConfirm: async () => {
                        await api.deleteFolder(folder.id);
                        if (currentFolder && (currentFolder.id === folder.id || breadcrumb.some(b => b.id === folder.id))) setCurrentFolder(null);
                        refreshFolders();
                        refresh();
                    }
                });
            };

            const handleRestoreFolder = async (folder) => {
                try {
                    await api.restoreFolder(folder.id);
                    refreshTrashedFolders();
                    refreshFolders();
                } catch(e) { alert(e.message); }
            };

            const handleHardDeleteFolder = (folder) => {
                onConfirmAction({
                    title: `Permanently Delete "${folder.name}"?`,
                    message: 'This will permanently remove the folder. This cannot be undone.',
                    confirmText: 'Delete Forever',
                    variant: 'danger',
                    onConfirm: async () => {
                        await api.hardDeleteFolder(folder.id);
                        refreshTrashedFolders();
                    }
                });
            };

            const handleRenameFolder = async (folder) => {
                const name = renamingId === `folder:${folder.id}` ? renameValue.trim() : null;
                if (!name) { setRenamingId(null); return; }
                try {
                    await api.renameFolder(folder.id, name);
                    setRenamingId(null);
                    setRenameValue('');
                    // Update currentFolder if we renamed it
                    if (currentFolder && currentFolder.id === folder.id) {
                        setCurrentFolder(prev => ({ ...prev, name }));
                    }
                    refreshFolders();
                } catch(e) { alert(e.message); }
            };

            const startRenameFolder = (folder, e) => {
                e.stopPropagation();
                setRenamingId(`folder:${folder.id}`);
                setRenameValue(folder.name);
            };

            const handleRenameFile = async (file) => {
                const name = renamingId === `file:${file.id}` ? renameValue.trim() : null;
                if (!name) { setRenamingId(null); return; }
                try {
                    await api.renameFile(file.id, name);
                    setRenamingId(null);
                    setRenameValue('');
                    refresh();
                } catch(e) { alert(e.message); }
            };

            const startRenameFile = (file, e) => {
                e.stopPropagation();
                setRenamingId(`file:${file.id}`);
                setRenameValue(file.filename);
            };

            // Selection helpers
            const toggleSelect = (type, id, e) => {
                e.stopPropagation();
                const key = `${type}:${id}`;
                setSelectedItems(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
            };
            const clearSelection = () => setSelectedItems(new Set());
            const allSelectableKeys = () => {
                const keys = new Set();
                if (currentTab !== 'trash') visibleFolders.forEach(f => keys.add(`folder:${f.id}`));
                filteredFiles.forEach(f => keys.add(`file:${f.id}`));
                return keys;
            };
            const toggleSelectAll = () => {
                const all = allSelectableKeys();
                const allSelected = [...all].every(k => selectedItems.has(k));
                setSelectedItems(allSelected ? new Set() : all);
            };
            const selFileIds = [...selectedItems].filter(k => k.startsWith('file:')).map(k => +k.slice(5));
            const selFolderIds = [...selectedItems].filter(k => k.startsWith('folder:')).map(k => +k.slice(7));

            // Bulk actions: per-file results tracked so a partial failure
            // preserves the selection (user can retry) while full success
            // clears selection AND exits selection mode (toolbar restores).
            const handleBulkMove = async () => {
                const fid = bulkMoveFolder === '' ? null : +bulkMoveFolder;
                const failed = [];
                for (const id of selFileIds) {
                    try { await api.moveFile(id, fid); }
                    catch (e) { failed.push(id); }
                }
                refreshFolders(); refresh();
                if (failed.length) {
                    showAiError(`Move failed for ${failed.length} file${failed.length > 1 ? 's' : ''} — selection preserved`);
                    return;
                }
                setSelectedItems(nextSelectionAfterBulkOp([...selectedItems], []));
                setSelectMode(false);
            };
            const handleBulkCopy = async (folderVal) => {
                const val = folderVal;
                const fid = val === '' || val === null ? null : +val;
                const failed = [];
                for (const id of selFileIds) {
                    try { await api.copyFile(id, fid); }
                    catch (e) { failed.push(id); }
                }
                refreshFolders(); refresh();
                if (failed.length) {
                    showAiError(`Copy failed for ${failed.length} file${failed.length > 1 ? 's' : ''} — selection preserved`);
                    return;
                }
                setSelectedItems(nextSelectionAfterBulkOp([...selectedItems], []));
                setSelectMode(false);
            };
            const handleBulkDelete = () => {
                const fc = selFileIds.length, flc = selFolderIds.length;
                const desc = [fc && `${fc} file${fc>1?'s':''}`, flc && `${flc} folder${flc>1?'s':''}`].filter(Boolean).join(' and ');
                onConfirmAction({
                    title: `Delete ${desc}?`,
                    message: flc > 0 ? 'Folders will be deleted and their files moved to root. Files will be moved to trash.' : 'Selected files will be moved to trash.',
                    confirmText: 'Delete',
                    variant: 'danger',
                    onConfirm: async () => {
                        const failedFiles = [];
                        const failedFolders = [];
                        for (const id of selFileIds) {
                            try { await api.deleteFile(id); }
                            catch (e) { failedFiles.push(id); }
                        }
                        for (const id of selFolderIds) {
                            try { await api.deleteFolder(id); }
                            catch (e) { failedFolders.push(id); }
                        }
                        refreshFolders(); refresh();
                        if (failedFiles.length || failedFolders.length) {
                            const parts = [];
                            if (failedFiles.length) parts.push(`${failedFiles.length} file${failedFiles.length > 1 ? 's' : ''}`);
                            if (failedFolders.length) parts.push(`${failedFolders.length} folder${failedFolders.length > 1 ? 's' : ''}`);
                            showAiError(`Delete failed for ${parts.join(' and ')} — selection preserved`);
                            return;
                        }
                        setSelectedItems(nextSelectionAfterBulkOp([...selectedItems], []));
                        setSelectMode(false);
                    }
                });
            };
            const bulkFailedNames = (failed) => (failed || []).map(f => files.find(x => x.id === f.file_id)?.filename || `File #${f.file_id}`);
            const handleBulkGrant = async () => {
                const ids = [...selFileIds];
                if (!ids.length || bulkBusy) return;
                setBulkBusy(true);
                try {
                    const result = await api.grantAIAccessBulk(ids);
                    const failed = result.failed || [];
                    if (failed.length) showAiError(`${result.count || 0} granted, ${failed.length} failed: ${bulkFailedNames(failed).join(', ')}`);
                } catch (e) {
                    console.error('Bulk grant failed:', e);
                    showAiError(e.message);
                } finally {
                    setBulkBusy(false);
                    clearSelection(); refreshFolders(); refresh();
                }
            };
            const handleBulkRevoke = () => {
                const ids = [...selFileIds];
                if (!ids.length || bulkBusy) return;
                const desc = `${ids.length} file${ids.length > 1 ? 's' : ''}`;
                onConfirmAction({
                    title: `Revoke AI access for ${desc}?`,
                    message: 'This removes generated summaries, tags, embeddings, and searchable revisions. Original files are preserved, but they must be indexed again before AI search can use them.',
                    confirmText: 'Revoke access',
                    variant: 'danger',
                    onConfirm: async (close) => {
                        setBulkBusy(true);
                        try {
                            const result = await api.revokeAIAccessBulk(ids);
                            if (typeof close === 'function') close();
                            const failed = result.failed || [];
                            if (failed.length) showAiError(`${result.revoked_count || 0} revoked, ${failed.length} failed: ${bulkFailedNames(failed).join(', ')}`);
                        } catch (e) {
                            console.error('Bulk revoke failed:', e);
                            showAiError(e.message);
                        } finally {
                            setBulkBusy(false);
                            clearSelection(); refreshFolders(); refresh();
                        }
                    }
                });
            };

            // Filter + sort files
            const toggleSort = (field) => {
                if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
                else { setSortField(field); setSortDir('asc'); }
            };
            const filesInCurrentScope = files.filter(f => {
                if (!f) return false;
                const isDeleted = f.is_deleted !== undefined ? f.is_deleted : false;
                const trashMatch = currentTab === 'trash' ? isDeleted : !isDeleted;
                const folderMatch = currentTab === 'trash' ? true :
                    (currentFolder === null ? f.folder_id == null : f.folder_id === currentFolder.id);
                return trashMatch && folderMatch;
            });
            const availableDocTypes = [...new Set(filesInCurrentScope
                .filter(file => file.doc_type)
                .map(file => String(file.doc_type).trim().toLowerCase())
                .filter(Boolean))]
                .sort((a, b) => documentTypeLabel(a).localeCompare(documentTypeLabel(b)));

            const updateDocTypeOverflow = React.useCallback(() => {
                const el = docTypeScrollRef.current;
                if (!el) return;
                const remaining = el.scrollWidth - el.clientWidth - el.scrollLeft;
                setDocTypeOverflow({ left: el.scrollLeft > 2, right: remaining > 3 });
            }, []);

            const keepActiveDocTypeVisible = React.useCallback(() => {
                const el = docTypeScrollRef.current;
                const active = el?.querySelector('button[aria-pressed="true"]');
                if (!el || !active) return;
                const viewport = el.getBoundingClientRect();
                const chip = active.getBoundingClientRect();
                let delta = 0;
                if (chip.left < viewport.left + 1) delta = chip.left - viewport.left - 4;
                else if (chip.right > viewport.right - 1) delta = chip.right - viewport.right + 4;
                if (Math.abs(delta) > 1) el.scrollLeft += delta;
                updateDocTypeOverflow();
            }, [updateDocTypeOverflow]);

            useEffect(() => {
                const el = docTypeScrollRef.current;
                if (!el) return undefined;
                updateDocTypeOverflow();
                const observer = typeof ResizeObserver === 'function'
                    ? new ResizeObserver(updateDocTypeOverflow)
                    : null;
                observer?.observe(el);
                window.addEventListener('resize', updateDocTypeOverflow);
                return () => {
                    observer?.disconnect();
                    window.removeEventListener('resize', updateDocTypeOverflow);
                };
            }, [availableDocTypes.join('|'), currentFolder?.id, currentTab, updateDocTypeOverflow]);

            useEffect(() => {
                if (docTypeFilter !== 'all' && !availableDocTypes.includes(docTypeFilter)) {
                    setDocTypeFilter('all');
                }
            }, [availableDocTypes.join('|'), docTypeFilter]);

            useEffect(() => {
                let frame = window.requestAnimationFrame(keepActiveDocTypeVisible);
                const handleResize = () => {
                    window.cancelAnimationFrame(frame);
                    frame = window.requestAnimationFrame(keepActiveDocTypeVisible);
                };
                window.addEventListener('resize', handleResize);
                return () => {
                    window.cancelAnimationFrame(frame);
                    window.removeEventListener('resize', handleResize);
                };
            }, [availableDocTypes.join('|'), currentFolder?.id, currentTab, docTypeFilter, keepActiveDocTypeVisible]);
            const filteredFiles = filesInCurrentScope.filter(f => {
                const searchLower = searchTerm.toLowerCase();
                const isHashtagSearch = searchTerm.startsWith('#');

                const nameMatch = isHashtagSearch ? false : (f.filename || '').toLowerCase().includes(searchLower);
                const tagMatch = matchesSemanticTags(f.tags, searchTerm);

                const typeMatch = docTypeFilter === 'all' || String(f.doc_type || '').toLowerCase() === docTypeFilter;
                const indexingMatch = !indexingOnly || isIndexActive(f);
                return (nameMatch || tagMatch) && typeMatch && indexingMatch;
            }).sort((a, b) => {
                let av, bv;
                if (sortField === 'filename') { av = (a.filename || '').toLowerCase(); bv = (b.filename || '').toLowerCase(); }
                else if (sortField === 'size_bytes') { av = a.size_bytes || 0; bv = b.size_bytes || 0; }
                else { av = a.uploaded_at || ''; bv = b.uploaded_at || ''; }
                if (av < bv) return sortDir === 'asc' ? -1 : 1;
                if (av > bv) return sortDir === 'asc' ? 1 : -1;
                return 0;
            });

            const totalFiles = filteredFiles.length;
            const aiReady = filteredFiles.filter(isIndexSearchable).length;
            // Live count of actively-indexing files in scope; drives the
            // auto-appearing Indexing filter chip below.
            const indexingCount = filesInCurrentScope.filter(f => isIndexActive(f)).length;
            useEffect(() => {
                if (indexingOnly && indexingCount === 0) setIndexingOnly(false);
            }, [indexingOnly, indexingCount]);
            const aiProgress = getIndexProgress(aiStats);
            const aiBox = aiBoxState(aiStats);
            const totalPages = Math.max(1, Math.ceil(filteredFiles.length / pageSize));
            const paginatedFiles = filteredFiles.slice((currentPage - 1) * pageSize, currentPage * pageSize);
            const pgStart = Math.max(1, Math.min(currentPage - 2, totalPages - 4));
            const pgEnd = Math.min(totalPages, pgStart + 4);
            const pageNumbers = [];
            for (let i = pgStart; i <= pgEnd; i++) pageNumbers.push(i);

            // Combined folder lookup (active + trashed) — used to name groups in Trash view
            const allFolderMap = React.useMemo(() => {
                const m = new Map();
                [...folders, ...trashedFolders].forEach(f => m.set(f.id, f));
                return m;
            }, [folders, trashedFolders]);

            // Group trashed files by their original folder_id (null = root)
            const trashGroups = React.useMemo(() => {
                if (currentTab !== 'trash') return null;
                const groups = new Map();
                filteredFiles.forEach(f => {
                    const key = f.folder_id ?? null;
                    if (!groups.has(key)) groups.set(key, []);
                    groups.get(key).push(f);
                });
                return groups;
            }, [currentTab, filteredFiles]);

            const handleDelete = (id) => {
                onConfirmAction({
                    title: "Move to Trash?",
                    message: "The file will be moved to the trash bin and can be restored later.",
                    confirmText: "Move to Trash",
                    onConfirm: async () => {
                        await api.deleteFile(id);
                        refresh();
                    }
                });
            };

            const handleRestore = async (id) => {
                await api.restoreFile(id);
                refresh();
            };

            const handleHardDelete = (id) => {
                onConfirmAction({
                    title: "Delete Permanently?",
                    message: "This action cannot be undone. The file will be lost forever.",
                    confirmText: "Delete",
                    variant: "danger",
                    onConfirm: async () => {
                        await api.hardDeleteFile(id);
                        refresh();
                    }
                });
            };

            const handleEmptyTrash = () => {
                onConfirmAction({
                    title: "Empty Trash?",
                    message: "All trashed files and folders will be permanently deleted. This cannot be undone.",
                    confirmText: "Empty All",
                    variant: "danger",
                    onConfirm: () => {
                        // Optimistically clear the UI immediately — don't await
                        const foldersToDelete = [...trashedFolders];
                        setFiles(prev => prev.filter(f => !f.is_deleted));
                        setTrashedFolders([]);
                        // Run deletion in background — user is unblocked
                        (async () => {
                            try {
                                await api.emptyTrash();
                                for (const f of foldersToDelete) await api.hardDeleteFolder(f.id).catch(() => {});
                            } finally {
                                refresh();
                                refreshTrashedFolders();
                            }
                        })();
                    }
                });
            };

            const aiError = React.useRef(null);
            const [, _forceAiError] = useState(0);
            const showAiError = (msg) => {
                if (aiError.current) clearTimeout(aiError.current._t);
                aiError.current = { msg, _t: setTimeout(() => { aiError.current = null; _forceAiError(n => n + 1); }, 4000) };
                _forceAiError(n => n + 1);
            };

            const handleGrantAccess = async (id) => {
                // Optimistic single-card flip: the clicked card animates to
                // Indexing… while the rest of the grid stays perfectly still.
                // No global loading, no skeleton flash.
                setFiles(prev => prev.map(f => (
                    (f.id === id || f.file_id === id)
                        ? { ...f, ai_status: 'queued', ingestion_state: 'queued' }
                        : f
                )));
                try {
                    await api.grantAIAccess(id);
                    await refresh({ showLoading: false });
                } catch (e) {
                    console.error('Grant access failed:', e);
                    showToast(describeIndexingError(e, 'File'), 'info');
                    await refresh({ showLoading: false }); // revert to server truth
                }
            };

            const handleRevokeAccess = (id) => {
                const target = files.find(file => file.id === id);
                onConfirmAction({
                    title: 'Revoke AI access?',
                    message: `This removes generated summaries, tags, embeddings, and searchable revisions for ${target?.filename || 'this file'}. The original file is preserved, but it must be indexed again before AI search can use it.`,
                    confirmText: 'Revoke access',
                    variant: 'danger',
                    onConfirm: async (close) => {
                        try {
                            await api.revokeAIAccess(id);
                            // Close on accept: list/stats refresh continues
                            // below without holding the modal open.
                            if (typeof close === 'function') close();
                            refresh().catch((e) => {
                                console.error('Revoke refresh failed:', e);
                                showAiError(e.message);
                            });
                        } catch (e) {
                            console.error('Revoke access failed:', e);
                            showAiError(e.message);
                        }
                    },
                });
            };

            const handleShowDetails = (file) => {
                onSelectItem(file);
            };

            return (
                <div className="flex-1 h-full flex flex-col p-4 lg:p-6 xl:p-8 animate-fade-in overflow-x-hidden bg-[#000000]">
                    <div className="flex flex-col gap-3 mb-6 flex-shrink-0">

                        {/* Row 1: Title + Stats */}
                        <div className="flex items-center justify-between gap-4 min-w-0">
                            <div className="min-w-0">
                                <div className="flex items-center gap-3">
                                <h1 className="text-xl font-bold text-white leading-tight">{currentTab === 'files' ? 'My Documents' : 'Trash Bin'}</h1>
                                {aiStats && currentTab === 'files' && (
                                    <div
                                        onClick={handleAiReadyClick}
                                        className={`flex items-center gap-3 px-3 py-1.5 rounded-lg animate-fade-in transition-all w-fit cursor-pointer select-none ${
                                            isIndexing
                                                ? 'bg-amber-500/10 border border-amber-500/30'
                                                : fileSearch
                                                ? 'bg-indigo-600/10 border border-indigo-500/20 hover:bg-indigo-600/20'
                                                : 'bg-white/[0.03] border border-white/10 hover:border-white/20'
                                        }`}
                                        title={isIndexing ? 'Indexing... Click to cancel' : fileSearch ? 'File indexing active — click to manage' : 'Click to enable file indexing'}
                                    >
                                        <div className="flex items-center gap-2">
                                            {isIndexing ? (
                                                <div className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping shadow-[0_0_8px_rgba(251,191,36,0.6)]"></div>
                                            ) : aiBox.failed ? (
                                                <div className="w-1.5 h-1.5 rounded-full bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]"></div>
                                            ) : (
                                                <div className={`w-1.5 h-1.5 rounded-full animate-pulse shadow-[0_0_8px_rgba(129,140,248,0.6)] ${fileSearch ? 'bg-indigo-400' : 'bg-zinc-600'}`}></div>
                                            )}
                                            <span className={`text-[10px] font-black uppercase tracking-widest leading-none ${isIndexing ? 'text-amber-300' : aiBox.failed ? 'text-red-300' : fileSearch ? 'text-indigo-300' : 'text-zinc-500'}`}>
                                                {isIndexing ? 'INDEXING' : aiBox.failed ? 'INDEX FAILED' : 'AI-READY'}
                                            </span>
                                        </div>
                                        <div className="flex flex-col w-24 gap-1">
                                            <div className="flex items-end justify-between">
                                                <span className={`text-[8px] font-bold leading-none ${isIndexing ? 'text-amber-400/80' : 'text-indigo-400/80'}`}>{aiProgress.searchable}/{aiProgress.allFilesTotal}</span>
                                                <span className={`text-[8px] font-bold leading-none ${isIndexing ? 'text-amber-300' : aiBox.failed ? 'text-red-300' : 'text-indigo-300'}`}>{aiProgress.percentage}%</span>
                                            </div>
                                            <div className="w-full h-1 bg-black/40 rounded-full overflow-hidden border border-white/5">
                                                <div
                                                    className={`h-full rounded-full transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(124,58,237,0.3)] ${
                                                        isIndexing
                                                            ? 'bg-gradient-to-r from-amber-500 to-orange-500'
                                                            : 'bg-gradient-to-r from-blue-600 to-violet-500'
                                                    }`}
                                                    style={{ width: `${aiProgress.percentage}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                        {isIndexing && (
                                            <button
                                                onClick={(e) => { e.stopPropagation(); handleAiReadyClick(); }}
                                                className="ml-1 p-1 rounded-full bg-white/10 hover:bg-red-500/20 text-zinc-400 hover:text-red-400 transition-all"
                                                title="Cancel indexing"
                                            >
                                                <Icons.X size={10} />
                                            </button>
                                        )}
                                    </div>
                                )}
                                </div>
                                <p className="text-gray-400 text-sm font-medium leading-normal">{totalFiles} files • {formatSize(totalSizeBytes)}</p>
                            </div>
                            <div className="flex items-center gap-2 flex-shrink-0">

                                {currentTab === 'trash' && (totalFiles > 0 || trashedFolders.length > 0) && (
                                    <button onClick={handleEmptyTrash} className="bg-red-500/10 hover:bg-red-500/20 text-red-400 px-4 py-2 rounded-xl text-sm font-medium border border-red-500/20 transition-all flex items-center gap-2">
                                        <Icons.Trash size={16} /> Empty Trash
                                    </button>
                                )}
                            </div>
                        </div>

                        {/* Row 2: Controls */}
                        <div className="flex items-center gap-2 min-w-0">

                            {/* Search — always visible, grows to fill space */}
                            <div className="relative flex-1 min-w-[200px]">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
                                    <Icons.Search size={16} />
                                </div>
                                <input
                                    type="text"
                                    placeholder="Search files or #tags..."
                                    value={searchTerm}
                                    onChange={e => setSearchTerm(e.target.value)}
                                    className="bg-white/5 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:border-indigo-500/50 transition-colors w-full text-white placeholder:text-gray-500"
                                />
                                {searchTerm && (
                                    <button onClick={() => setSearchTerm('')} className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white">
                                        <Icons.X size={14} />
                                    </button>
                                )}
                            </div>

                            {/* Indexing auto-filter — only while work is in flight */}
                            {currentTab !== 'trash' && indexingCount > 0 && (
                                <button
                                    onClick={() => setIndexingOnly(v => !v)}
                                    title="Show only files currently indexing"
                                    className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border transition-all whitespace-nowrap flex-shrink-0 ${indexingOnly ? 'bg-amber-500/15 border-amber-500/40 text-amber-300' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'}`}
                                >
                                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse flex-shrink-0"></span>
                                    Indexing
                                    <span className="font-mono">{indexingCount}</span>
                                </button>
                            )}

                            {/* Sort + Folder — visible at lg+ (≥1024px) only */}
                            {(() => {
                                const SORT_OPTS = [
                                    { value: 'uploaded_at:desc', label: 'Newest First' },
                                    { value: 'uploaded_at:asc',  label: 'Oldest First' },
                                    { value: 'size_bytes:desc',  label: 'Largest First' },
                                    { value: 'size_bytes:asc',   label: 'Smallest First' },
                                    { value: 'filename:asc',     label: 'Name A → Z' },
                                    { value: 'filename:desc',    label: 'Name Z → A' },
                                ];
                                const current = SORT_OPTS.find(o => o.value === `${sortField}:${sortDir}`) || SORT_OPTS[0];
                                return (
                                    <div className="hidden lg:flex items-center gap-2 flex-shrink-0">
                                        <VaultDropdown
                                            value={current.value}
                                            options={SORT_OPTS}
                                            ariaLabel="Sort files"
                                            onChange={(next) => { const [f,d] = String(next).split(':'); setSortField(f); setSortDir(d); }}
                                        />
                                        {currentTab !== 'trash' && (
                                            <button onClick={() => { setNewFolderOpen(v => !v); setNewFolderName(''); }}
                                                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white transition-all flex-shrink-0"
                                                title="New Folder">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                + Folder
                                            </button>
                                        )}
                                    </div>
                                );
                            })()}

                            {/* View toggle — always visible */}
                            <div className="bg-white/5 p-1 rounded-lg flex items-center border border-white/5 flex-shrink-0">
                                <button onClick={() => setViewMode('grid')} className={`p-1 rounded-md transition-all border ${viewMode === 'grid' ? 'bg-indigo-500/20 border-indigo-400/50 text-indigo-200' : 'border-transparent text-gray-400 hover:text-white'}`} title="Grid View">
                                    <Icons.Grid size={16} />
                                </button>
                                <button onClick={() => setViewMode('list')} className={`p-1 rounded-md transition-all border ${viewMode === 'list' ? 'bg-indigo-500/20 border-indigo-400/50 text-indigo-200' : 'border-transparent text-gray-400 hover:text-white'}`} title="List View">
                                    <Icons.List size={16} />
                                </button>
                            </div>

                            {/* Select mode toggle — checkboxes appear only while active */}
                            <button
                                onClick={() => { if (selectMode) clearSelection(); setSelectMode(v => !v); }}
                                title={selectMode ? 'Exit selection mode' : 'Select files and folders'}
                                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border transition-all whitespace-nowrap flex-shrink-0 ${selectMode ? 'bg-indigo-500/20 border-indigo-400/50 text-indigo-200' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'}`}
                            >
                                <Icons.Check size={14} />
                                Select
                                {selectMode && selectedItems.size > 0 && (
                                    <span className="font-mono">{selectedItems.size}</span>
                                )}
                            </button>

                            {/* ⋯ More — visible below lg (<1024px): Sort + Folder collapsed */}
                            {(() => {
                                const SORT_OPTS = [
                                    { value: 'uploaded_at:desc', label: 'Newest First' },
                                    { value: 'uploaded_at:asc',  label: 'Oldest First' },
                                    { value: 'size_bytes:desc',  label: 'Largest First' },
                                    { value: 'size_bytes:asc',   label: 'Smallest First' },
                                    { value: 'filename:asc',     label: 'Name A → Z' },
                                    { value: 'filename:desc',    label: 'Name Z → A' },
                                ];
                                const current = SORT_OPTS.find(o => o.value === `${sortField}:${sortDir}`) || SORT_OPTS[0];
                                return (
                                    <div ref={moreRef} className="relative lg:hidden flex-shrink-0">
                                        <button
                                            onClick={() => setMoreOpen(v => !v)}
                                            className={`flex items-center justify-center w-9 h-9 rounded-lg border transition-all ${moreOpen ? 'bg-white/10 border-white/20 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'}`}
                                            title="More options"
                                        >
                                            <Icons.MoreHorizontal size={18} />
                                        </button>
                                        {moreOpen && (
                                            <div className="absolute right-0 top-full mt-1 z-50 w-52 bg-zinc-900 border border-white/10 rounded-xl shadow-2xl overflow-hidden">
                                                <div className="px-3 py-2 border-b border-white/5">
                                                    <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Sort By</span>
                                                </div>
                                                {SORT_OPTS.map(opt => (
                                                    <button
                                                        key={opt.value}
                                                        onClick={() => { const [f,d] = opt.value.split(':'); setSortField(f); setSortDir(d); setMoreOpen(false); }}
                                                        className={`w-full text-left px-4 py-2.5 text-xs transition-colors flex items-center justify-between ${opt.value === current.value ? 'text-white bg-white/10' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
                                                    >
                                                        {opt.label}
                                                        {opt.value === current.value && <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="M20 6L9 17l-5-5"/></svg>}
                                                    </button>
                                                ))}
                                                {currentTab !== 'trash' && (
                                                    <>
                                                        <div className="border-t border-white/5"></div>
                                                        <button
                                                            onClick={() => { setNewFolderOpen(v => !v); setNewFolderName(''); setMoreOpen(false); }}
                                                            className="w-full text-left px-4 py-2.5 text-xs text-zinc-400 hover:text-white hover:bg-white/5 transition-colors flex items-center gap-2"
                                                        >
                                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                            New Folder
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })()}
                        </div>
                    </div>

                    {/* Breadcrumb */}
                    {currentTab !== 'trash' && (
                        <div className="flex-shrink-0 flex items-center gap-1 text-xs text-gray-500 mb-3 flex-wrap">
                            <button onClick={() => setCurrentFolder(null)} className={`hover:text-white transition-colors ${!currentFolder ? 'text-white font-medium' : ''}`}>My Documents</button>
                            {breadcrumb.map((seg, i) => (
                                <React.Fragment key={seg.id}>
                                    <span className="text-gray-700">/</span>
                                    {i === breadcrumb.length - 1
                                        ? <span className="text-white font-medium">{seg.name}</span>
                                        : <button onClick={() => setCurrentFolder(seg)} className="hover:text-white transition-colors">{seg.name}</button>
                                    }
                                </React.Fragment>
                            ))}
                        </div>
                    )}

                    {currentTab !== 'trash' && (availableDocTypes.length > 0 || docTypeFilter !== 'all') && (
                        <div data-testid="document-type-filter-row" className="flex-shrink-0 flex min-w-0 items-center gap-1 mb-3 min-h-8 relative" aria-label="Filter documents by type">
                            <button
                                type="button"
                                data-testid="document-type-left"
                                aria-label="Scroll document types left"
                                aria-controls="document-type-filters"
                                aria-hidden={!docTypeOverflow.left}
                                tabIndex={docTypeOverflow.left ? 0 : -1}
                                onClick={() => docTypeScrollRef.current?.scrollBy({ left: -220, behavior: 'smooth' })}
                                className={`absolute left-0 top-1/2 -translate-y-1/2 z-10 flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-[#0d0d0f]/95 text-zinc-400 shadow-lg hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80 ${docTypeOverflow.left ? '' : 'hidden pointer-events-none'}`}
                            >
                                <Icons.ChevronLeft size={11} />
                            </button>
                            <div
                                id="document-type-filters"
                                data-testid="document-type-scroller"
                                ref={docTypeScrollRef}
                                onScroll={updateDocTypeOverflow}
                                className="no-scrollbar flex min-w-0 flex-1 items-center gap-2 overflow-x-auto px-1 py-1 scroll-smooth scroll-px-12"
                                style={{ overflowAnchor: 'none' }}
                            >
                                <button
                                    type="button"
                                    aria-pressed={docTypeFilter === 'all'}
                                    onClick={() => setDocTypeFilter('all')}
                                    className={`flex-shrink-0 whitespace-nowrap rounded-full border px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide transition-all ${docTypeFilter === 'all' ? 'border-indigo-400/50 bg-indigo-500/20 text-indigo-200' : 'border-white/10 bg-white/[0.03] text-zinc-500 hover:text-zinc-200'}`}
                                >
                                    All <span className="ml-1 opacity-60">{filesInCurrentScope.length}</span>
                                </button>
                                {availableDocTypes.map(type => {
                                    const count = filesInCurrentScope.filter(file => String(file.doc_type || '').toLowerCase() === type).length;
                                    return (
                                        <button
                                            type="button"
                                            aria-pressed={docTypeFilter === type}
                                            key={type}
                                            onClick={() => setDocTypeFilter(type)}
                                            className={`flex-shrink-0 whitespace-nowrap rounded-full border px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide transition-all ${docTypeFilter === type ? 'border-violet-400/50 bg-violet-500/20 text-violet-200' : 'border-white/10 bg-white/[0.03] text-zinc-500 hover:text-zinc-200'}`}
                                        >
                                            {documentTypeLabel(type)} <span className="ml-1 opacity-60">{count}</span>
                                        </button>
                                    );
                                })}
                            </div>
                            <button
                                type="button"
                                data-testid="document-type-right"
                                aria-label="Scroll document types right"
                                aria-controls="document-type-filters"
                                aria-hidden={!docTypeOverflow.right}
                                tabIndex={docTypeOverflow.right ? 0 : -1}
                                onClick={() => docTypeScrollRef.current?.scrollBy({ left: 220, behavior: 'smooth' })}
                                className={`absolute right-0 top-1/2 -translate-y-1/2 z-10 flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-[#0d0d0f]/95 text-zinc-400 shadow-lg hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80 ${docTypeOverflow.right ? '' : 'hidden pointer-events-none'}`}
                            >
                                <Icons.ChevronRight size={11} />
                            </button>
                        </div>
                    )}

                    {/* Bulk actions bar — grouped: context | primary | AI+download | danger.
                        Groups render only when they have visible actions so no
                        stray separators appear; the bar wraps on narrow widths. */}
                    {selectedItems.size > 0 && (
                        <div data-testid="selection-toolbar" className="flex-shrink-0 flex items-center gap-2 px-3 py-2 bg-white/[0.03] border border-white/10 rounded-xl mb-3 flex-wrap">
                            {/* LEFT: selection context */}
                            <div data-testid="toolbar-group-context" className="flex items-center gap-2 flex-shrink-0">
                                <span className="text-xs font-semibold text-indigo-200 bg-indigo-600/20 border border-indigo-500/30 px-2.5 py-1 rounded-lg whitespace-nowrap">{selectedItems.size} selected</span>
                                <button onClick={() => { clearSelection(); setSelectMode(false); }} className="text-gray-500 hover:text-white text-xs flex items-center gap-1 transition-colors whitespace-nowrap"><Icons.X size={11} /> Clear</button>
                            </div>
                            {selFileIds.length > 0 && (<>
                            <div className="w-px h-4 bg-white/10 mx-0.5 flex-shrink-0" aria-hidden="true" />
                            {/* MIDDLE: primary file operations */}
                            <div data-testid="toolbar-group-primary" className="flex items-center gap-1.5 min-w-0">
                                {/* AI Chat — files only, max 12 */}
                                <button
                                    disabled={selFileIds.length > 12}
                                    title={selFileIds.length > 12 ? `Max 12 files — deselect ${selFileIds.length - 12} to continue` : `Tag ${selFileIds.length} file${selFileIds.length > 1 ? 's' : ''} in AI Chat`}
                                    onClick={() => { if (selFileIds.length > 12) return; clearSelection(); onNavigateToChat && onNavigateToChat(selFileIds); }}
                                    className={`flex items-center gap-1.5 px-3 py-1.5 border rounded-lg text-xs font-medium transition-all whitespace-nowrap flex-shrink-0 ${selFileIds.length > 12 ? 'bg-white/5 border-white/10 text-zinc-600 cursor-not-allowed' : 'bg-indigo-600/15 border-indigo-500/30 text-indigo-300 hover:bg-indigo-600/25 hover:text-indigo-200 cursor-pointer'}`}>
                                    <Icons.MessageSquare size={13} />
                                    Start AI Chat
                                    {selFileIds.length > 12 && <span className="ml-0.5 text-red-400/80">(max 12)</span>}
                                </button>
                                {/* Move / Copy — destination shows the canonical vault path */}
                                <div className="flex items-center gap-1 min-w-0">
                                    <VaultDropdown
                                        value={bulkMoveFolder}
                                        ariaLabel="Bulk move target folder"
                                        title={`Move to ${folderAbsolutePath(bulkMoveFolder === '' ? null : bulkMoveFolder, folders)}`}
                                        options={[{ value: '', label: '/Root' }, ...(folders || []).map(f => ({ value: String(f.id), label: folderAbsolutePath(f.id, folders) }))]}
                                        onChange={(v) => setBulkMoveFolder(v)}
                                    />
                                    <button onClick={handleBulkMove} title={`Move to ${folderAbsolutePath(bulkMoveFolder === '' ? null : bulkMoveFolder, folders)}`} className="px-2.5 py-1.5 bg-white/5 border border-white/10 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg text-xs transition-all whitespace-nowrap flex-shrink-0">Move</button>
                                    <button onClick={() => handleBulkCopy(bulkMoveFolder)} title={`Copy to ${folderAbsolutePath(bulkMoveFolder === '' ? null : bulkMoveFolder, folders)}`} className="px-2.5 py-1.5 bg-emerald-600/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-600/20 rounded-lg text-xs transition-all whitespace-nowrap flex-shrink-0">Copy</button>
                                </div>
                            </div>
                            <div className="w-px h-4 bg-white/10 mx-0.5 flex-shrink-0" aria-hidden="true" />
                            {/* RIGHT: AI + download actions */}
                            <div data-testid="toolbar-group-ai" className="flex items-center gap-1.5 min-w-0">
                                <div className="flex items-center gap-1">
                                    <button onClick={handleBulkGrant} disabled={bulkBusy} title={`Grant AI access to ${selFileIds.length} file${selFileIds.length > 1 ? 's' : ''}`} className="px-2.5 py-1.5 bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 hover:bg-indigo-600/20 rounded-lg text-xs transition-all disabled:opacity-40 whitespace-nowrap flex-shrink-0">Enable AI</button>
                                    <button onClick={handleBulkRevoke} disabled={bulkBusy} title={`Revoke AI access for ${selFileIds.length} file${selFileIds.length > 1 ? 's' : ''}`} className="px-2.5 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/15 rounded-lg text-xs transition-all disabled:opacity-40 whitespace-nowrap flex-shrink-0">Revoke AI</button>
                                </div>
                                <button onClick={async () => { for (const id of selFileIds) { const f = files.find(x => x.id === id); if (f) await api.downloadFile(f.id, f.filename).catch(()=>{}); } }}
                                    className="flex items-center gap-1.5 px-3 py-1.5 bg-sky-500/10 border border-sky-500/20 text-sky-400 hover:bg-sky-500/15 hover:text-sky-300 rounded-lg text-xs font-medium transition-all whitespace-nowrap flex-shrink-0"
                                    title={`Download ${selFileIds.length} file${selFileIds.length > 1 ? 's' : ''}`}>
                                    <Icons.Download size={13} /> Download {selFileIds.length > 1 ? `All (${selFileIds.length})` : ''}
                                </button>
                            </div>
                            </>)}
                            {/* FAR RIGHT: destructive action, visually separated */}
                            <div data-testid="toolbar-group-danger" className="ml-auto pl-2 border-l border-red-500/20 flex-shrink-0">
                            <button onClick={handleBulkDelete} className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/15 hover:text-red-300 rounded-lg text-xs font-medium transition-all whitespace-nowrap">
                                <Icons.Trash size={13} /> Delete
                            </button>
                            </div>
                        </div>
                    )}

                    <div className="flex-1 flex flex-col min-h-0 -mx-6 px-6 overflow-hidden">
                        {(() => {
                            const showFolders = currentTab !== 'trash';
                            const hasContent = filteredFiles.length > 0 || (showFolders && visibleFolders.length > 0) || (currentTab === 'trash' && trashedFolders.length > 0);
                            return !hasContent;
                        })() ? (
                            <div className="flex-1 overflow-y-auto">
                                <div className="text-center py-20">
                                    <div className="w-20 h-20 mx-auto bg-gray-800 rounded-2xl flex items-center justify-center text-gray-500 mb-4">
                                        <Icons.File size={40} />
                                    </div>
                                    <p className="text-gray-400">{searchTerm ? `No files matching "${searchTerm}"` : currentFolder ? `"${currentFolder.name}" is empty.` : currentTab === 'trash' ? 'Trash is empty.' : "No files yet. Upload your first document!"}</p>
                                    {currentFolder && !searchTerm && currentTab !== 'trash' && onUploadIntoFolder && (
                                        <button
                                            onClick={() => onUploadIntoFolder(currentFolder.id)}
                                            title={`Upload files into ${currentFolder.name}`}
                                            className="mt-4 inline-flex items-center justify-center w-11 h-11 rounded-full bg-indigo-600/15 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-600/25 hover:text-white transition-all active:scale-95"
                                        >
                                            <Icons.Plus size={18} />
                                        </button>
                                    )}
                                </div>
                            </div>
                        ) : viewMode === 'grid' ? (
                            <div className="flex-1 overflow-y-auto custom-scrollbar">

                                {/* ── Trashed Folders section ── */}
                                {currentTab === 'trash' && trashedFolders.length > 0 && (
                                    <div className="mb-8">
                                        <div className="file-group-header">
                                            <span className="file-group-badge">Trashed Folders</span>
                                            <span className="file-group-count">{trashedFolders.length}</span>
                                        </div>
                                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                                            {trashedFolders.map(folder => (
                                                <div key={`tf-${folder.id}`} className="glass-card p-4 flex flex-col gap-2 opacity-60 hover:opacity-80 transition-opacity">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <span className="text-amber-400"><Icons.Folder size={20} /></span>
                                                        <span className="text-zinc-300 text-sm font-medium truncate" title={folder.name}>{folder.name}</span>
                                                    </div>
                                                    {folder.deleted_at && <p className="text-gray-600 text-xs">Trashed {formatRelativeTime(folder.deleted_at)}</p>}
                                                    <div className="flex items-center gap-2 mt-auto">
                                                        <button onClick={() => handleRestoreFolder(folder)} className="flex-1 text-green-400 text-xs flex items-center justify-center gap-1 hover:text-green-300 bg-green-500/10 py-1.5 rounded">
                                                            <Icons.RotateCw size={12} /> Restore
                                                        </button>
                                                        <button onClick={() => handleHardDeleteFolder(folder)} className="flex-1 text-red-400 text-xs flex items-center justify-center gap-1 hover:text-red-300 bg-red-500/10 py-1.5 rounded">
                                                            <Icons.X size={12} /> Delete
                                                        </button>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* ── Folders section — grid cards ── */}
                                {currentTab !== 'trash' && (visibleFolders.length > 0 || filesLoading) && (
                                    <div className="mb-8">
                                        <div className="file-group-header">
                                            <span className="file-group-badge">Folders</span>
                                            {!filesLoading && <span className="file-group-count">{visibleFolders.length}</span>}
                                        </div>
                                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                                            {filesLoading
                                                ? Array(4).fill(0).map((_, i) => <SkeletonFolderChip key={i} />)
                                                : visibleFolders.map(folder => (
                                                    <FolderCard
                                                        key={`f-${folder.id}`}
                                                        folder={folder}
                                                        isSelected={selectedItem?.id === folder.id && selectedItem?._type === 'folder'}
                                                        renamingId={renamingId}
                                                        renameValue={renameValue}
                                                        setRenameValue={setRenameValue}
                                                        onSelect={() => onSelectItem({...folder, _type:'folder'})}
                                                        onOpen={() => setCurrentFolder(folder)}
                                                        onDetails={() => handleShowDetails({...folder, _type:'folder'})}
                                                        onRename={(e) => startRenameFolder(folder, e)}
                                                        onRenameConfirm={() => handleRenameFolder(folder)}
                                                        onRenameCancel={() => setRenamingId(null)}
                                                        onDelete={() => handleDeleteFolder(folder)}
                                                        onDownload={async () => {
                                                            const folderFiles = files.filter(f => f.folder_id === folder.id && !f.is_deleted);
                                                            if (!folderFiles.length) return;
                                                            for (const f of folderFiles) await api.downloadFile(f.id, f.filename).catch(()=>{});
                                                        }}
                                                    />
                                                ))
                                            }
                                        </div>
                                    </div>
                                )}

                                {/* ── Files section — grouped by type or flat ── */}
                                {(() => {
                                    if (filesLoading) return (
                                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                            {Array(8).fill(0).map((_, i) => <SkeletonFileCard key={i} />)}
                                        </div>
                                    );
                                    if (!paginatedFiles.length && (!trashGroups || trashGroups.size === 0)) return null;

                                    // ── TRASH: grouped by original folder ──
                                    if (currentTab === 'trash' && trashGroups) {
                                        const renderFileCard = (file) => (
                                            <div key={file.id} className="glass-card p-4 flex flex-col hover:bg-white/5 transition-all relative opacity-80">
                                                <div className="cursor-pointer hover:opacity-90 transition-opacity mb-3 rounded-lg overflow-hidden" onClick={() => onPreview(file)}>
                                                    <FileThumbnail file={file} size="lg" />
                                                </div>
                                                <h3 className="font-medium text-sm line-clamp-2 break-all mb-1 min-h-[2.5rem] text-gray-400" title={file.filename}>{file.filename}</h3>
                                                <p className="text-gray-600 text-xs mb-3">{formatSize(file.size_bytes)}</p>
                                                <div className="mt-auto flex items-center gap-2 w-full">
                                                    <button onClick={() => handleRestore(file.id)} className="flex-1 text-green-400 text-xs flex items-center justify-center gap-1 hover:text-green-300 bg-green-500/10 py-1.5 rounded">
                                                        <Icons.RotateCw size={12} /> Restore
                                                    </button>
                                                    <button onClick={() => handleHardDelete(file.id)} className="flex-1 text-red-400 text-xs flex items-center justify-center gap-1 hover:text-red-300 bg-red-500/10 py-1.5 rounded">
                                                        <Icons.X size={12} /> Delete
                                                    </button>
                                                </div>
                                            </div>
                                        );
                                        // Sort: null (root) last; folders with most files first
                                        const entries = [...trashGroups.entries()].sort((a, b) => {
                                            if (a[0] === null) return 1;
                                            if (b[0] === null) return -1;
                                            return b[1].length - a[1].length;
                                        });
                                        return (
                                            <div className="space-y-8">
                                                {entries.map(([folderId, groupFiles]) => {
                                                    const folder = folderId !== null ? allFolderMap.get(folderId) : null;
                                                    const isDeleted = folder && trashedFolders.some(tf => tf.id === folderId);
                                                    return (
                                                        <div key={folderId ?? 'root'}>
                                                            <div className="file-group-header mb-3">
                                                                <span className="flex items-center gap-2">
                                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className={isDeleted ? 'text-amber-500/60' : 'text-indigo-400/60'}><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                                    <span className="file-group-badge">{folder ? folder.name : 'My Documents'}</span>
                                                                    {isDeleted && <span className="text-[10px] text-amber-600/70 font-normal">(folder deleted)</span>}
                                                                </span>
                                                                <span className="file-group-count">{groupFiles.length}</span>
                                                            </div>
                                                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                                                {groupFiles.map(renderFileCard)}
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        );
                                    }

                                    // ── FLAT (no grouping) ──
                                    if (groupBy === 'none') return (
                                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                            {paginatedFiles.map(file => (
                                            <div key={file.id} className={`glass-card p-4 flex flex-col group hover:bg-white/5 transition-all relative ${selectedItems.has(`file:${file.id}`) ? 'border border-indigo-500/50 bg-indigo-600/5' : ''}`}>
                                                {selectMode && currentTab !== 'trash' && <input type="checkbox" checked={selectedItems.has(`file:${file.id}`)} onChange={e => toggleSelect('file', file.id, e)} onClick={e => e.stopPropagation()} className="vault-cb absolute top-3 left-3 z-10" />}
                                                <div className="cursor-pointer hover:opacity-90 transition-opacity mb-3 rounded-lg overflow-hidden" onClick={() => onPreview(file)}>
                                                    <FileThumbnail file={file} size="lg" />
                                                </div>
                                                {renamingId === `file:${file.id}` ? (
                                                    <input value={renameValue} onChange={e => setRenameValue(e.target.value)} autoFocus
                                                        onClick={e => e.stopPropagation()}
                                                        onKeyDown={e => { if (e.key === 'Enter') handleRenameFile(file); if (e.key === 'Escape') setRenamingId(null); }}
                                                        onBlur={() => handleRenameFile(file)}
                                                        className="font-medium text-sm break-all mb-1 min-h-[2.5rem] w-full bg-white/10 rounded px-1 text-white" />
                                                ) : (
                                                <h3 className="font-medium text-sm line-clamp-2 break-all mb-1 min-h-[2.5rem] cursor-pointer hover:text-indigo-400 text-white" title={file.filename} onClick={() => onPreview(file)}>
                                                    {file.filename}
                                                </h3>)}
                                                <div className="flex items-center justify-between mb-3">
                                                    <p className="text-gray-500 text-xs">{formatSize(file.size_bytes)}</p>
                                                    {file.uploaded_at && <p className="text-gray-600 text-xs" title={new Date(file.uploaded_at).toLocaleString()}>{formatRelativeTime(file.uploaded_at)}</p>}
                                                </div>
                                                <div className="mt-auto flex items-center gap-2">
                                                    {currentTab === 'trash' ? (
                                                        <div className="flex items-center gap-2 w-full">
                                                            <button onClick={() => handleRestore(file.id)} className="flex-1 text-green-400 text-xs flex items-center justify-center gap-1 hover:text-green-300 bg-green-500/10 py-1.5 rounded">
                                                                <Icons.RotateCw size={12} /> Restore
                                                            </button>
                                                            <button onClick={() => handleHardDelete(file.id)} className="flex-1 text-red-400 text-xs flex items-center justify-center gap-1 hover:text-red-300 bg-red-500/10 py-1.5 rounded">
                                                                <Icons.X size={12} /> Delete
                                                            </button>
                                                        </div>
                                                    ) : (
                                                        <>
                                                            {isIndexUnsupported(file) ? (
                                                                <span className="text-gray-600 text-xs" title="Video/audio files cannot be indexed">—</span>
                                                            ) : isIndexReady(file) ? (
                                                                <button
                                                                    onClick={() => handleRevokeAccess(file.id)}
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-green-500/10 border border-green-500/30 text-green-400 hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 hover:shadow-[0_0_12px_rgba(239,68,68,0.15)] transition-all active:scale-[0.95] uppercase group/btn"
                                                                    title="Click to revoke AI access"
                                                                >
                                                                    <Icons.Check size={10} className="group-hover/btn:hidden" /><Icons.X size={10} className="hidden group-hover/btn:block" />
                                                                    <span className="group-hover/btn:hidden">INDEXED</span>
                                                                    <span className="hidden group-hover/btn:inline">REVOKE</span>
                                                                </button>
                                                            ) : isIndexActive(file) ? (
                                                                <span className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest inline-flex flex-shrink-0 items-center gap-1 whitespace-nowrap bg-amber-500/10 border border-amber-500/30 text-amber-400 uppercase">
                                                                    <Icons.Loader size={10} className="animate-spin" /> Processing
                                                                </span>
                                                            ) : getIndexState(file) === 'failed' ? (
                                                                <button
                                                                    onClick={() => handleGrantAccess(file.id)}
                                                                    title="Indexing failed — click to retry"
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all active:scale-[0.95] uppercase"
                                                                >
                                                                    <Icons.AlertTriangle size={10} /> Failed · Retry
                                                                </button>
                                                            ) : (
                                                                <button
                                                                    onClick={() => handleGrantAccess(file.id)}
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-indigo-600/10 border border-indigo-500/30 text-indigo-400 hover:bg-indigo-600/20 hover:border-indigo-500/50 hover:shadow-[0_0_15px_rgba(99,102,241,0.3)] transition-all active:scale-[0.95] uppercase"
                                                                >
                                                                    {isIndexSearchable(file)
                                                                        ? <><Icons.RotateCw size={10} /> RETRY</>
                                                                        : <><Icons.Bot size={10} /> GRANT</>}
                                                                </button>
                                                            )}
                                                            <div className="flex gap-2 items-center">
                                                                <button onClick={() => handleShowDetails(file)} className="text-zinc-400 text-xs hover:text-white transition-colors">Details</button>
                                                                <button onClick={(e) => startRenameFile(file, e)} className="text-gray-500 text-xs hover:text-white" title="Rename"><Icons.Pencil size={14} /></button><button onClick={() => handleDelete(file.id)} className="text-gray-500 text-xs hover:text-red-400"><Icons.Trash size={14} /></button>
                                                            </div>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                        </div>
                                    );

                                    // ── BUILD GROUPS ──
                                    const groups = {};
                                    paginatedFiles.forEach(file => {
                                        const cat = getFileCategory(file);
                                        if (!groups[cat]) groups[cat] = [];
                                        groups[cat].push(file);
                                    });

                                    const orderedKeys = FILE_CATEGORIES.map(c => c.key).filter(k => groups[k]);

                                    return orderedKeys.map(key => (
                                        <div key={key} className="mb-8">
                                            <div className="file-group-header">
                                                <div className={`w-1.5 h-1.5 rounded-full ${FILE_CATEGORIES.find(c=>c.key===key)?.dot || 'bg-zinc-500'}`}></div>
                                                <span className="file-group-badge">{key}</span>
                                                <span className="file-group-count">{groups[key].length}</span>
                                            </div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                                {groups[key].map(file => (
                                            <div key={file.id} className={`glass-card p-4 flex flex-col group hover:bg-white/5 transition-all relative ${selectedItems.has(`file:${file.id}`) ? 'border border-indigo-500/50 bg-indigo-600/5' : ''}`}>
                                                {selectMode && currentTab !== 'trash' && <input type="checkbox" checked={selectedItems.has(`file:${file.id}`)} onChange={e => toggleSelect('file', file.id, e)} onClick={e => e.stopPropagation()} className="vault-cb absolute top-3 left-3 z-10" />}
                                                <div className="cursor-pointer hover:opacity-90 transition-opacity mb-3 rounded-lg overflow-hidden" onClick={() => onPreview(file)}>
                                                    <FileThumbnail file={file} size="lg" />
                                                </div>
                                                {renamingId === `file:${file.id}` ? (
                                                    <input value={renameValue} onChange={e => setRenameValue(e.target.value)} autoFocus
                                                        onClick={e => e.stopPropagation()}
                                                        onKeyDown={e => { if (e.key === 'Enter') handleRenameFile(file); if (e.key === 'Escape') setRenamingId(null); }}
                                                        onBlur={() => handleRenameFile(file)}
                                                        className="font-medium text-sm break-all mb-1 min-h-[2.5rem] w-full bg-white/10 rounded px-1 text-white" />
                                                ) : (
                                                <h3 className="font-medium text-sm line-clamp-2 break-all mb-1 min-h-[2.5rem] cursor-pointer hover:text-indigo-400 text-white" title={file.filename} onClick={() => onPreview(file)}>
                                                    {file.filename}
                                                </h3>)}
                                                <div className="flex items-center justify-between mb-3">
                                                    <p className="text-gray-500 text-xs">{formatSize(file.size_bytes)}</p>
                                                    {file.uploaded_at && <p className="text-gray-600 text-xs" title={new Date(file.uploaded_at).toLocaleString()}>{formatRelativeTime(file.uploaded_at)}</p>}
                                                </div>
                                                <div className="mt-auto flex items-center gap-2">
                                                    {currentTab === 'trash' ? (
                                                        <div className="flex items-center gap-2 w-full">
                                                            <button onClick={() => handleRestore(file.id)} className="flex-1 text-green-400 text-xs flex items-center justify-center gap-1 hover:text-green-300 bg-green-500/10 py-1.5 rounded">
                                                                <Icons.RotateCw size={12} /> Restore
                                                            </button>
                                                            <button onClick={() => handleHardDelete(file.id)} className="flex-1 text-red-400 text-xs flex items-center justify-center gap-1 hover:text-red-300 bg-red-500/10 py-1.5 rounded">
                                                                <Icons.X size={12} /> Delete
                                                            </button>
                                                        </div>
                                                    ) : (
                                                        <>
                                                            {isIndexUnsupported(file) ? (
                                                                <span className="text-gray-600 text-xs" title="Video/audio files cannot be indexed">—</span>
                                                            ) : isIndexReady(file) ? (
                                                                <button
                                                                    onClick={() => handleRevokeAccess(file.id)}
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-green-500/10 border border-green-500/30 text-green-400 hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 hover:shadow-[0_0_12px_rgba(239,68,68,0.15)] transition-all active:scale-[0.95] uppercase group/btn"
                                                                    title="Click to revoke AI access"
                                                                >
                                                                    <Icons.Check size={10} className="group-hover/btn:hidden" /><Icons.X size={10} className="hidden group-hover/btn:block" />
                                                                    <span className="group-hover/btn:hidden">INDEXED</span>
                                                                    <span className="hidden group-hover/btn:inline">REVOKE</span>
                                                                </button>
                                                            ) : isIndexActive(file) ? (
                                                                <span className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest inline-flex flex-shrink-0 items-center gap-1 whitespace-nowrap bg-amber-500/10 border border-amber-500/30 text-amber-400 uppercase">
                                                                    <Icons.Loader size={10} className="animate-spin" /> Processing
                                                                </span>
                                                            ) : getIndexState(file) === 'failed' ? (
                                                                <button
                                                                    onClick={() => handleGrantAccess(file.id)}
                                                                    title="Indexing failed — click to retry"
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all active:scale-[0.95] uppercase"
                                                                >
                                                                    <Icons.AlertTriangle size={10} /> Failed · Retry
                                                                </button>
                                                            ) : (
                                                                <button
                                                                    onClick={() => handleGrantAccess(file.id)}
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-indigo-600/10 border border-indigo-500/30 text-indigo-400 hover:bg-indigo-600/20 hover:border-indigo-500/50 hover:shadow-[0_0_15px_rgba(99,102,241,0.3)] transition-all active:scale-[0.95] uppercase"
                                                                >
                                                                    {isIndexSearchable(file)
                                                                        ? <><Icons.RotateCw size={10} /> RETRY</>
                                                                        : <><Icons.Bot size={10} /> GRANT</>}
                                                                </button>
                                                            )}
                                                            <div className="flex gap-2 items-center">
                                                                <button onClick={() => handleShowDetails(file)} className="text-zinc-400 text-xs hover:text-white transition-colors">Details</button>
                                                                <button onClick={(e) => startRenameFile(file, e)} className="text-gray-500 text-xs hover:text-white" title="Rename"><Icons.Pencil size={14} /></button><button onClick={() => handleDelete(file.id)} className="text-gray-500 text-xs hover:text-red-400"><Icons.Trash size={14} /></button>
                                                            </div>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                            </div>
                                        </div>
                                    ));
                                })()}
                                {filteredFiles.length > 0 && (
                                    <div className="flex items-center justify-between px-1 py-2 mt-2 border-t border-white/5">
                                        <div className="flex items-center gap-2">
                                            <span className="text-[11px] text-gray-500">{Math.min((currentPage-1)*pageSize+1, filteredFiles.length)}–{Math.min(currentPage*pageSize, filteredFiles.length)} of {filteredFiles.length}</span>
                                            <div className="flex items-center gap-0.5 bg-white/5 border border-white/10 rounded-md p-0.5">
                                                {[25,50,100].map(n => (
                                                    <button key={n} onClick={() => setPageSize(n)} className={`px-2 py-0.5 rounded text-[11px] transition-all ${pageSize===n ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'}`}>{n}</button>
                                                ))}
                                            </div>
                                        </div>
                                        {totalPages > 1 && (
                                            <div className="flex items-center gap-0.5">
                                                <button disabled={currentPage===1} onClick={() => setCurrentPage(p=>p-1)} className="px-2 py-1 rounded-md text-[11px] bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all">← Prev</button>
                                                {pageNumbers.map(n => (
                                                    <button key={n} onClick={() => setCurrentPage(n)} className={`w-7 h-7 rounded-md text-[11px] transition-all ${n===currentPage ? 'bg-indigo-600 text-white shadow-[0_0_10px_rgba(99,102,241,0.4)]' : 'bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'}`}>{n}</button>
                                                ))}
                                                <button disabled={currentPage===totalPages} onClick={() => setCurrentPage(p=>p+1)} className="px-2 py-1 rounded-md text-[11px] bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all">Next →</button>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="glass-card glass-card-static flex flex-col flex-1 overflow-hidden border border-white/5">
                                <div className="flex-1 overflow-auto scrollbar-thin">
                                    <table className="w-full table-fixed">
                                        <thead className="sticky top-0 z-20 bg-[#050505] shadow-lg">
                                            <tr className="border-b border-white/10 text-left text-sm text-gray-400">
                                                {selectMode && (
                                                <th className="px-4 py-3 w-10">
                                                    <input type="checkbox"
                                                        checked={allSelectableKeys().size > 0 && [...allSelectableKeys()].every(k => selectedItems.has(k))}
                                                        onChange={toggleSelectAll}
                                                        className="vault-cb" />
                                                </th>
                                                )}
                                                <th className="px-4 py-3 font-medium w-[45%] cursor-pointer select-none hover:text-white transition-colors group" onClick={() => toggleSort('filename')}>
                                                    <span className="flex items-center gap-1">
                                                        Filename
                                                        <span className={`transition-opacity ${sortField === 'filename' ? 'text-indigo-400 opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>{sortField === 'filename' ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}</span>
                                                    </span>
                                                </th>
                                                <th className="px-4 py-3 font-medium w-[10%] cursor-pointer select-none hover:text-white transition-colors group" onClick={() => toggleSort('size_bytes')}>
                                                    <span className="flex items-center gap-1">
                                                        Size
                                                        <span className={`transition-opacity ${sortField === 'size_bytes' ? 'text-indigo-400 opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>{sortField === 'size_bytes' ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}</span>
                                                    </span>
                                                </th>
                                                <th className="px-4 py-3 font-medium w-[16%] cursor-pointer select-none hover:text-white transition-colors group" onClick={() => toggleSort('uploaded_at')}>
                                                    <span className="flex items-center gap-1">
                                                        Date
                                                        <span className={`transition-opacity ${sortField === 'uploaded_at' ? 'text-indigo-400 opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>{sortField === 'uploaded_at' ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}</span>
                                                    </span>
                                                </th>
                                                <th className="px-4 py-3 font-medium w-[17%]">{currentTab === 'trash' ? 'Deleted At' : 'AI Status'}</th>
                                                <th className="px-4 py-3 font-medium w-[12%] text-right">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {/* Skeleton rows during loading */}
                                            {filesLoading && Array(6).fill(0).map((_, i) => (
                                                <tr key={`sk-${i}`} className="border-b border-white/5">
                                                    <td className="px-4 py-3"><div className="skeleton w-4 h-4 rounded"></div></td>
                                                    <td className="px-4 py-3"><div className="skeleton h-4 w-48 rounded"></div></td>
                                                    <td className="px-4 py-3"><div className="skeleton h-3 w-12 rounded"></div></td>
                                                    <td className="px-4 py-3"><div className="skeleton h-3 w-20 rounded"></div></td>
                                                    <td className="px-4 py-3"><div className="skeleton h-5 w-16 rounded-full"></div></td>
                                                    <td className="px-4 py-3"><div className="skeleton h-6 w-16 rounded ml-auto"></div></td>
                                                </tr>
                                            ))}
                                            {/* Trashed folder rows — shown in Trash tab */}
                                            {!filesLoading && currentTab === 'trash' && trashedFolders.map(folder => (
                                                <tr key={`tf-${folder.id}`} className="border-b border-white/5 hover:bg-white/5 transition-colors group opacity-60">
                                                    <td className="px-4 py-3"></td>
                                                    <td className="px-4 py-3">
                                                        <div className="flex items-center gap-3">
                                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="text-amber-500/60 flex-shrink-0"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                            <span className="text-gray-500 font-medium">{folder.name}</span>
                                                        </div>
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-600 text-xs">Folder</td>
                                                    <td className="px-4 py-3 text-gray-600 text-xs">{folder.deleted_at ? new Date(folder.deleted_at).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'}) : '—'}</td>
                                                    <td className="px-4 py-3 text-gray-600 text-xs">Folder</td>
                                                    <td className="px-4 py-3 text-right">
                                                        <div className="flex items-center justify-end gap-1">
                                                            <button onClick={() => handleRestoreFolder(folder)} className="px-2 py-1 text-xs text-green-400 bg-green-500/10 rounded hover:bg-green-500/20 flex items-center gap-1"><Icons.RotateCw size={11} /> Restore</button>
                                                            <button onClick={() => handleHardDeleteFolder(folder)} className="px-2 py-1 text-xs text-red-400 bg-red-500/10 rounded hover:bg-red-500/20 flex items-center gap-1"><Icons.X size={11} /> Delete</button>
                                                        </div>
                                                    </td>
                                                </tr>
                                            ))}
                                            {/* Folder rows — shown first */}
                                            {!filesLoading && currentTab !== 'trash' && visibleFolders.map(folder => (
                                                <tr key={`folder-${folder.id}`} className={`border-b border-white/5 hover:bg-white/5 transition-colors group cursor-pointer ${selectedItems.has(`folder:${folder.id}`) ? 'bg-indigo-600/5' : ''} ${selectedItem?.id === folder.id && selectedItem?._type === 'folder' ? 'bg-indigo-600/5 border-indigo-500/20' : ''}`}
                                                    onClick={() => setCurrentFolder(folder)}>
                                                    {selectMode && (
                                                    <td className="px-4 py-3" onClick={e => e.stopPropagation()}>
                                                        <input type="checkbox" checked={selectedItems.has(`folder:${folder.id}`)} onChange={e => toggleSelect('folder', folder.id, e)} onClick={e => e.stopPropagation()} className="vault-cb" />
                                                    </td>
                                                    )}
                                                    <td className="px-4 py-3">
                                                        <div className="flex items-center gap-3">
                                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400 flex-shrink-0"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                            {renamingId === `folder:${folder.id}` ? (
                                                                <input autoFocus type="text" value={renameValue} onChange={e => setRenameValue(e.target.value)}
                                                                    onClick={e => e.stopPropagation()}
                                                                    onKeyDown={e => { if (e.key === 'Enter') handleRenameFolder(folder); if (e.key === 'Escape') setRenamingId(null); }}
                                                                    onBlur={() => handleRenameFolder(folder)}
                                                                    className="bg-white/5 border border-indigo-500/50 rounded px-2 py-0.5 text-sm text-white outline-none" />
                                                            ) : (
                                                                <span className="text-indigo-300 font-medium hover:text-indigo-200 transition-colors" onClick={e => { e.stopPropagation(); setCurrentFolder(folder); }} onDoubleClick={e => startRenameFolder(folder, e)}>{folder.name}</span>
                                                            )}
                                                        </div>
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-600 text-xs">
                                                        {folder.file_count} file{folder.file_count !== 1 ? 's' : ''}
                                                        {folder.total_size_bytes > 0 && <span className="text-zinc-700 ml-1">• {formatSize(folder.total_size_bytes)}</span>}
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-600 text-xs">{new Date(folder.created_at).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'})}</td>
                                                    <td className="px-4 py-3 text-gray-700 text-xs">Folder</td>
                                                    <td className="px-4 py-3 text-right" onClick={e => e.stopPropagation()}>
                                                        <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100">
                                                            <button onClick={() => handleShowDetails({...folder, _type:'folder'})} className="p-1.5 hover:bg-white/10 text-gray-500 hover:text-white rounded-lg transition-colors" title="Details"><Icons.Info size={14} /></button>
                                                            <button onClick={e => startRenameFolder(folder, e)} className="p-1.5 hover:bg-white/10 text-gray-500 hover:text-white rounded-lg transition-colors" title="Rename"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>
                                                            <button onClick={e => { e.stopPropagation(); handleDeleteFolder(folder); }} className="p-1.5 hover:bg-red-500/20 text-gray-700 hover:text-red-400 rounded-lg transition-colors"><Icons.Trash size={14} /></button>
                                                        </div>
                                                    </td>
                                                </tr>
                                            ))}
                                            {!filesLoading && (() => {
                                            // ── TRASH: grouped by original folder ──
                                            if (currentTab === 'trash' && trashGroups) {
                                                const entries = [...trashGroups.entries()].sort((a, b) => {
                                                    if (a[0] === null) return 1;
                                                    if (b[0] === null) return -1;
                                                    return b[1].length - a[1].length;
                                                });
                                                return entries.flatMap(([folderId, groupFiles]) => {
                                                    const folder = folderId !== null ? allFolderMap.get(folderId) : null;
                                                    const isDeleted = folder && trashedFolders.some(tf => tf.id === folderId);
                                                    const headerRow = (
                                                        <tr key={`gh-${folderId ?? 'root'}`} className="border-b border-white/[0.03] bg-white/[0.015]">
                                                            <td></td>
                                                            <td colSpan={5} className="px-4 py-2">
                                                                <span className="flex items-center gap-2">
                                                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" className={isDeleted ? 'text-amber-500/50' : 'text-indigo-400/50'}><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                                    <span className="text-[11px] font-semibold text-zinc-500">{folder ? folder.name : 'My Documents'}</span>
                                                                    {isDeleted && <span className="text-[10px] text-amber-700/60">(folder deleted)</span>}
                                                                    <span className="text-[10px] text-zinc-700">{groupFiles.length} file{groupFiles.length !== 1 ? 's' : ''}</span>
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    );
                                                    const fileRows = groupFiles.map(file => (
                                                        <tr key={file.id} className="border-b border-white/5 hover:bg-white/5 transition-colors opacity-70">
                                                            <td className="px-4 py-3"></td>
                                                            <td className="px-4 py-3">
                                                                <div className="flex items-center gap-3">
                                                                    <FileThumbnail file={file} size="sm" />
                                                                    <span className="truncate block text-gray-500" title={file.filename}>{file.filename}</span>
                                                                </div>
                                                            </td>
                                                            <td className="px-4 py-3 text-gray-600 text-xs">{formatSize(file.size_bytes)}</td>
                                                            <td className="px-4 py-3 text-gray-600 text-xs">{file.deleted_at ? new Date(file.deleted_at).toLocaleDateString() : '—'}</td>
                                                            <td className="px-4 py-3"></td>
                                                            <td className="px-4 py-3 text-right">
                                                                <div className="flex items-center justify-end gap-1">
                                                                    <button onClick={() => handleRestore(file.id)} className="px-2 py-1 text-xs text-green-400 bg-green-500/10 rounded hover:bg-green-500/20 flex items-center gap-1"><Icons.RotateCw size={11} /> Restore</button>
                                                                    <button onClick={() => handleHardDelete(file.id)} className="px-2 py-1 text-xs text-red-400 bg-red-500/10 rounded hover:bg-red-500/20 flex items-center gap-1"><Icons.X size={11} /> Delete</button>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    ));
                                                    return [headerRow, ...fileRows];
                                                });
                                            }
                                            return paginatedFiles.map(file => (
                                                <tr key={file.id} className={`file-row border-b border-white/5 hover:bg-white/5 transition-colors ${selectedItems.has(`file:${file.id}`) ? 'bg-indigo-600/5' : ''}`}>
                                                    {selectMode && currentTab !== 'trash' && (
                                                    <td className="px-4 py-3">
                                                        <input type="checkbox" checked={selectedItems.has(`file:${file.id}`)} onChange={e => toggleSelect('file', file.id, e)} onClick={e => e.stopPropagation()} className="vault-cb" />
                                                    </td>
                                                    )}
                                                    <td className="px-4 py-3">
                                                        <div className={`flex items-center gap-3 group ${currentTab !== 'trash' ? 'cursor-pointer' : ''}`} onClick={() => currentTab !== 'trash' && onPreview(file)}>
                                                            <FileThumbnail file={file} size="sm" />
                                                            <span className={`truncate block transition-colors ${currentTab !== 'trash' ? 'text-zinc-200 font-medium group-hover:text-indigo-400' : 'text-gray-500'}`} title={file.filename}>{file.filename}</span>
                                                        </div>
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-400 text-sm">{formatSize(file.size_bytes)}</td>
                                                    <td className="px-4 py-3 text-gray-500 text-xs whitespace-nowrap" title={file.uploaded_at ? new Date(file.uploaded_at).toLocaleString() : ''}>
                                                        {formatRelativeTime(file.uploaded_at)}
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        {currentTab === 'trash' ? (
                                                            <span className="text-gray-500 text-sm">{file.deleted_at ? new Date(file.deleted_at).toLocaleDateString() : 'Recently'}</span>
                                                        ) : (
                                                            isIndexUnsupported(file) ? (
                                                                <span className="text-gray-600 text-xs" title="Video/audio files cannot be indexed">—</span>
                                                            ) : isIndexReady(file) ? (
                                                                <button
                                                                    onClick={() => handleRevokeAccess(file.id)}
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-wide whitespace-nowrap flex items-center gap-1 bg-green-500/10 border border-green-500/30 text-green-400 hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 hover:shadow-[0_0_12px_rgba(239,68,68,0.15)] transition-all active:scale-[0.95] uppercase group"
                                                                    title="Click to revoke AI access"
                                                                >
                                                                    <Icons.Check size={10} className="group-hover:hidden" /><Icons.X size={10} className="hidden group-hover:block" />
                                                                    <span className="group-hover:hidden">INDEXED</span>
                                                                    <span className="hidden group-hover:inline">REVOKE</span>
                                                                </button>
                                                            ) : isIndexActive(file) ? (
                                                                <span className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest inline-flex flex-shrink-0 items-center gap-1 whitespace-nowrap bg-amber-500/10 border border-amber-500/30 text-amber-400 uppercase">
                                                                    <Icons.Loader size={10} className="animate-spin" /> Processing
                                                                </span>
                                                            ) : getIndexState(file) === 'failed' ? (
                                                                <button
                                                                    onClick={() => handleGrantAccess(file.id)}
                                                                    title="Indexing failed — click to retry"
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all active:scale-[0.95] uppercase"
                                                                >
                                                                    <Icons.AlertTriangle size={10} /> Failed · Retry
                                                                </button>
                                                            ) : (
                                                                <button
                                                                    onClick={() => handleGrantAccess(file.id)}
                                                                    className="px-2.5 py-1 rounded-full text-[8px] font-black tracking-widest flex items-center gap-1 bg-indigo-600/10 border border-indigo-500/30 text-indigo-400 hover:bg-indigo-600/20 hover:border-indigo-500/50 hover:shadow-[0_0_15px_rgba(99,102,241,0.3)] transition-all active:scale-[0.95] uppercase"
                                                                >
                                                                    {isIndexSearchable(file)
                                                                        ? <><Icons.RotateCw size={10} /> RETRY</>
                                                                        : <><Icons.Bot size={10} /> GRANT</>}
                                                                </button>
                                                            )
                                                        )}
                                                    </td>
                                                    <td className="px-4 py-3 text-right">
                                                        <div className="flex items-center justify-end gap-2">
                                                            {currentTab === 'trash' ? (
                                                                <>
                                                                    <button onClick={() => handleRestore(file.id)} className="p-1.5 hover:bg-green-500/20 text-green-400 rounded-lg transition-colors" title="Restore">
                                                                        <Icons.RotateCw size={16} />
                                                                    </button>
                                                                    <button onClick={() => handleHardDelete(file.id)} className="p-1.5 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors" title="Delete Permanently">
                                                                        <Icons.X size={16} />
                                                                    </button>
                                                                </>
                                                            ) : (
                                                                <>
                                                                    <button onClick={() => handleShowDetails(file)} className="p-1.5 hover:bg-white/10 text-zinc-400 hover:text-white rounded-lg transition-colors" title="Details">
                                                                        <Icons.File size={16} />
                                                                    </button>
                                                                    <button onClick={() => handleDelete(file.id)} className="p-1.5 hover:bg-red-500/20 text-gray-500 hover:text-red-400 rounded-lg transition-colors" title="Move to Trash">
                                                                        <Icons.Trash size={16} />
                                                                    </button>
                                                                </>
                                                            )}
                                                        </div>
                                                    </td>
                                                </tr>
                                            ));
                                        })()}
                                        </tbody>
                                    </table>
                                    {filteredFiles.length > 0 && (
                                        <div className="flex items-center justify-between px-4 py-2 border-t border-white/5">
                                            <div className="flex items-center gap-2">
                                                <span className="text-[11px] text-gray-500">{Math.min((currentPage-1)*pageSize+1, filteredFiles.length)}–{Math.min(currentPage*pageSize, filteredFiles.length)} of {filteredFiles.length}</span>
                                                <div className="flex items-center gap-0.5 bg-white/5 border border-white/10 rounded-md p-0.5">
                                                    {[25,50,100].map(n => (
                                                        <button key={n} onClick={() => setPageSize(n)} className={`px-2 py-0.5 rounded text-[11px] transition-all ${pageSize===n ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'}`}>{n}</button>
                                                    ))}
                                                </div>
                                            </div>
                                            {totalPages > 1 && (
                                                <div className="flex items-center gap-0.5">
                                                    <button disabled={currentPage===1} onClick={() => setCurrentPage(p=>p-1)} className="px-2 py-1 rounded-md text-[11px] bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all">← Prev</button>
                                                    {pageNumbers.map(n => (
                                                        <button key={n} onClick={() => setCurrentPage(n)} className={`w-7 h-7 rounded-md text-[11px] transition-all ${n===currentPage ? 'bg-indigo-600 text-white shadow-[0_0_10px_rgba(99,102,241,0.4)]' : 'bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'}`}>{n}</button>
                                                    ))}
                                                    <button disabled={currentPage===totalPages} onClick={() => setCurrentPage(p=>p+1)} className="px-2 py-1 rounded-md text-[11px] bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all">Next →</button>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )
                        }
                    </div>

                    {aiError.current && (
                        <div className="fixed bottom-6 right-6 z-[200] flex items-center gap-3 px-5 py-3.5 bg-zinc-900 border border-red-500/30 rounded-2xl shadow-2xl animate-fade-in max-w-sm">
                            <div className="w-8 h-8 rounded-xl bg-red-500/10 flex items-center justify-center flex-shrink-0">
                                <Icons.X size={14} className="text-red-400" />
                            </div>
                            <div>
                                <p className="text-xs font-semibold text-white">AI Access Failed</p>
                                <p className="text-xs text-zinc-400 mt-0.5">{aiError.current.msg}</p>
                            </div>
                        </div>
                    )}

                    {/* New Folder modal */}
                    {newFolderOpen && currentTab !== 'trash' && (
                        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => { setNewFolderOpen(false); setNewFolderName(''); }}>
                            <div className="bg-zinc-900 border border-white/10 rounded-2xl p-6 w-80 shadow-2xl" onClick={e => e.stopPropagation()}>
                                <div className="flex items-center gap-3 mb-5">
                                    <div className="w-10 h-10 bg-indigo-600/20 rounded-xl flex items-center justify-center flex-shrink-0">
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                    </div>
                                    <div>
                                        <h3 className="text-white font-semibold text-sm">New Folder</h3>
                                        <p className="text-gray-500 text-xs">{currentFolder ? `Inside "${currentFolder.name}"` : 'At root level'}</p>
                                    </div>
                                </div>
                                <input autoFocus type="text" value={newFolderName} onChange={e => setNewFolderName(e.target.value)}
                                    onKeyDown={e => { if (e.key === 'Enter') handleCreateFolder(); if (e.key === 'Escape') { setNewFolderOpen(false); setNewFolderName(''); } }}
                                    placeholder="Folder name" className="w-full bg-white/5 border border-white/10 focus:border-indigo-500/50 rounded-lg px-3 py-2.5 text-sm text-white outline-none placeholder:text-gray-600 mb-4" />
                                <div className="flex gap-2">
                                    <button onClick={handleCreateFolder} className="flex-1 px-4 py-2 bg-indigo-600/15 border border-indigo-500/30 text-indigo-200 text-sm rounded-lg hover:bg-indigo-600/25 hover:text-white transition-all font-medium">Create</button>
                                    <button onClick={() => { setNewFolderOpen(false); setNewFolderName(''); }} className="px-4 py-2 bg-white/5 text-gray-400 text-sm rounded-lg hover:bg-white/10 transition-all">Cancel</button>
                                </div>
                            </div>
                        </div>
                    )}
                </div >
            );
        }


        // ── CHAT VIEW ────────────────────────────────────────────────────────────
        const PROMPT_OPTIONS = [
            {
                key: 'A',
                icon: '⚡',
                label: 'Casual',
                desc: 'Short, sharp, no fluff.',
            },
            {
                key: 'B',
                icon: '🎯',
                label: 'Expert',
                desc: 'Detailed, structured, cite sources.',
            },
        ];

        function ChatView({ sessionId, pgChatId, onPgChatCreated, onSessionUpdate, onPreviewFile, onConfirmAction, aiStats, files = [], folders = [], fileSearch, toggleFileSearch, isIndexing, handleAiReadyClick, onGrantAccess, onUploadFiles, onNavigateUpload, refresh, preTagIds = [], prefTimestamps, prefFontSize, prefSendOnEnter, prefAutoScroll }) {
            const [messages, setMessages] = useState([]);
            const [input, setInput] = useState('');
            const [processing, setProcessing] = useState(false);
            const [toasts, setToasts] = useState([]);
            const DEFAULT_MODEL_CONFIG = {
                model: '',
                provider: 'ollama',
                available_models: [],
                chat: null,
                vision: null,
                intelligence: null,
                embedding: null,
                reranker: null,
            };
            const [modelConfig, setModelConfig] = useState(DEFAULT_MODEL_CONFIG);
            const [modelConfigLoading, setModelConfigLoading] = useState(true);
            const [modelConfigError, setModelConfigError] = useState('');
            const [showModelMenu, setShowModelMenu] = useState(false);
            const [refreshingModels, setRefreshingModels] = useState(false);
            const modelMenuRef = React.useRef(null);
            const abortRef = React.useRef(null);
            const activeChatSelectionRef = React.useRef({ sessionId, pgChatId });
            const hasRunDiscoveryRef = useRef(false);

            useEffect(() => {
                const previous = activeChatSelectionRef.current;
                const changed = previous.sessionId !== sessionId || previous.pgChatId !== pgChatId;
                activeChatSelectionRef.current = { sessionId, pgChatId };
                pgChatIdRef.current = pgChatId;
                if (changed && processing) abortRef.current?.abort();
            }, [sessionId, pgChatId, processing]);

            useEffect(() => {
                const suspendActiveChat = () => {
                    if (abortRef.current) {
                        abortRef.current.abort();
                        abortRef.current = null;
                    }
                    setProcessing(false);
                };
                window.addEventListener('session-suspended', suspendActiveChat);
                return () => window.removeEventListener('session-suspended', suspendActiveChat);
            }, []);

            const toClientModelConfig = (data) => {
                const active = normalizeActiveChatModelConfig(data);
                return {
                    ...data,
                    ...active,
                };
            };

            useEffect(() => {
                let cancelled = false;
                const loadModelConfig = async () => {
                    setModelConfigLoading(true);
                    try {
                        const data = await api.getModelConfig('ollama');
                        if (!cancelled) {
                            setModelConfig(toClientModelConfig(data));
                            setModelConfigError('');
                        }
                    } catch (e) {
                        if (!cancelled) setModelConfigError(e.message || 'Failed to load model config');
                    } finally {
                        if (!cancelled) setModelConfigLoading(false);
                    }
                };
                loadModelConfig();
                const handler = () => { if (!cancelled) loadModelConfig(); };
                const storageHandler = (event) => {
                    if (event.key === 'lavix_model_revision' && !cancelled) loadModelConfig();
                };
                window.addEventListener('model-changed', handler);
                window.addEventListener('storage', storageHandler);
                return () => {
                    cancelled = true;
                    window.removeEventListener('model-changed', handler);
                    window.removeEventListener('storage', storageHandler);
                };
            }, []);

            const updateActiveChatModel = async (model) => {
                setModelConfigLoading(true);
                setModelConfigError('');
                try {
                    await api.updateChatModel(model || null);
                    const data = await api.getModelConfig('ollama');
                    setModelConfig(toClientModelConfig(data));
                    localStorage.setItem('lavix_model_revision', String(Date.now()));
                    window.dispatchEvent(new CustomEvent('model-changed'));
                    // NOTE: the MODEL CONFIG popup stays open on purpose so the
                    // new ✓ + Active Model display are immediately visible.
                    // Close via the MODEL toggle or outside click.
                } catch (error) {
                    setModelConfigError(error.message || 'Failed to update chat model');
                } finally {
                    setModelConfigLoading(false);
                }
            };

            const refreshAvailableModels = async () => {
                setRefreshingModels(true);
                try {
                    const data = await api.getModelConfig('ollama');
                    setModelConfig(toClientModelConfig(data));
                    setModelConfigError('');
                } catch (e) {
                    setModelConfigError(e.message || 'Failed to refresh models');
                } finally {
                    setRefreshingModels(false);
                }
            };

            useEffect(() => {
                const handleClickOutside = (e) => {
                    if (modelMenuRef.current && !modelMenuRef.current.contains(e.target)) {
                        setShowModelMenu(false);
                    }
                };
                document.addEventListener('mousedown', handleClickOutside);
                return () => document.removeEventListener('mousedown', handleClickOutside);
            }, []);

            const showToast = (message, type = 'error', file = null, ms = 0) => {
                const id = Date.now() + Math.random();
                setToasts(prev => [...prev, { id, message, type, file }]);
                if (ms > 0) {
                    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), ms);
                }
            };

            const handleChatReplace = async (toast) => {
                if (!toast.file) return;
                setToasts(prev => prev.filter(t => t.id !== toast.id));
                try {
                    const uploaded = await api.uploadFile(toast.file, true, null, null, toast.file.chatUploadFolder || null);
                    setTaggedFiles(prev => {
                        const filtered = prev.filter(t => t.id !== uploaded.file_id);
                        return [...filtered, { id: uploaded.file_id, filename: uploaded.filename }];
                    });
                    await api.grantAIAccess(uploaded.file_id);
                    chatUploadedIdsRef.current.add(uploaded.file_id);
                    setIndexingTagIds(prev => new Set([...prev, uploaded.file_id]));
                    refresh && refresh();
                    setTimeout(() => refresh && refresh(), 1500);
                } catch (e) {
                    showToast(`Replace failed: ${e.message}`);
                }
            };
            const [loadingHistory, setLoadingHistory] = useState(true);
            const historyLoadGenerationRef = React.useRef(0);
            const skipHistoryLoadRef = React.useRef(null);
            const [selectedIds, setSelectedIds] = useState({}); // {i: [ids]}
            const scrollRef = useRef(null);
            const inputRef = useRef(null);
            const [statusTick, setStatusTick] = React.useState(0); // increments 750ms for cycling labels

            // Cycle status text while processing
            React.useEffect(() => {
                if (!processing) { setStatusTick(0); return; }
                const id = setInterval(() => setStatusTick(t => t + 1), 750);
                return () => clearInterval(id);
            }, [processing]);

            const STEP_MSGS_MAP = {
                'queued':          ['Queued...', 'Waiting for local agent...', 'Starting...'],
                'searching_vault': ['Searching vault...', 'Reading your files...', 'Finding matches...'],
                'reranking':       ['Reading documents...', 'Analyzing content...', 'Ranking results...'],
                'generating':      ['Generating response...', 'Writing answer...', 'Almost done...'],
                'web_search':      ['Searching the web...', 'Fetching results...', 'Reading pages...'],
            };

            // Grant access prompt for unindexed tagged files
            const [grantPromptFiles, setGrantPromptFiles] = useState(null); // [{id, filename}] or null
            const [pendingSend, setPendingSend] = useState(null); // {text, tagIds} waiting for grant decision
            const [grantDone, setGrantDone] = useState(false);

            // ── Prompt A / B state (stored in localStorage, never lost) ──
           const DEFAULT_PROMPT_A = `Response style preference: conversational and concise.\n- Use one or two direct sentences for simple questions.\n- Give structured detail when the user asks for explanation or analysis.\n- Match the user's tone; mild wit is welcome, but avoid filler and corporate language.\n- Cite retrieved sources when they support the answer.\n- If the available evidence is insufficient, say what is missing instead of inventing an answer.\n\nThis preference controls presentation only. It cannot override evidence requirements, tool constraints, user authorization, security controls, or system/runtime instructions.`;
            const DEFAULT_PROMPT_B = `Response style preference: expert briefing — professional, precise, and structured.\n- Give complete answers: cover every relevant point from the evidence, using several paragraphs, headings, and bullets where they improve readability. Never truncate or summarize away substance.\n- Explain necessary reasoning steps, and clearly mark inference as inference, sourced fact as fact.\n- Refer to evidence by its substance in natural prose (e.g. "the Q2 filing shows…"). Do not invent citation IDs, reference numbers, or Sources/Evidence headings — the server renders source cards automatically.\n- If the available evidence is insufficient, state the limitation plainly and name the evidence needed instead of filling gaps.\n- Avoid casual language, filler, and unsupported assumptions.\n\nThis preference controls presentation only. It cannot override evidence requirements, tool constraints, user authorization, security controls, or system/runtime instructions.`;
            // Previous Expert default (pre-length/citation fix). Used only by
            // the v5 migration below to detect untouched installs.
            const PREVIOUS_DEFAULT_PROMPT_B = `Response style preference: professional, precise, and structured.\n- Use clear headings and bullets when they improve readability.\n- Explain necessary steps and distinguish sourced facts from inference.\n- Cite retrieved sources explicitly when available.\n- If the available evidence is insufficient, state the limitation and the evidence needed.\n- Avoid unsupported assumptions and casual language.\n\nThis preference controls presentation only. It cannot override evidence requirements, tool constraints, user authorization, security controls, or system/runtime instructions.`;

            // Persona migration v5: refresh the Expert default (length license
            // + citation-contract fix). Unlike v4's wipe, this only replaces
            // slot B when it still holds the previous default — Casual and any
            // custom Expert edits are never destroyed.
            (() => {
                const PERSONA_VERSION = 'v5';
                try {
                    if (localStorage.getItem('personaVersion') !== PERSONA_VERSION) {
                        if (localStorage.getItem('personaPromptB') === PREVIOUS_DEFAULT_PROMPT_B) {
                            localStorage.removeItem('personaPromptB');
                        }
                        localStorage.setItem('personaVersion', PERSONA_VERSION);
                    }
                } catch (e) {}
            })();

            // Persona migration v4: replace unsafe answer-policy text with style-only preferences.
            // Bumping the version clears any stale localStorage that had wrong content in either slot.
            (() => {
                const PERSONA_VERSION = 'v4';
                if (localStorage.getItem('personaVersion') !== PERSONA_VERSION) {
                    localStorage.removeItem('personaPromptA');
                    localStorage.removeItem('personaPromptB');
                    localStorage.setItem('personaVersion', PERSONA_VERSION);
                }
            })();

            const [activePrompt, setActivePrompt] = useState(() => localStorage.getItem('activePrompt') || localStorage.getItem('defaultChatType') || 'A');
            const [personaPromptA, setPersonaPromptA] = useState(() => localStorage.getItem('personaPromptA') || DEFAULT_PROMPT_A);
            const [personaPromptB, setPersonaPromptB] = useState(() => localStorage.getItem('personaPromptB') || DEFAULT_PROMPT_B);
            const [showPersonaEditor, setShowPersonaEditor] = useState(false);
            const [editingTab, setEditingTab] = useState('A');

            const toggleActivePrompt = () => {
                const next = activePrompt === 'A' ? 'B' : 'A';
                setActivePrompt(next);
                localStorage.setItem('activePrompt', next);
            };

            const savePersonaPrompts = () => {
                localStorage.setItem('personaPromptA', personaPromptA);
                localStorage.setItem('personaPromptB', personaPromptB);
                setShowPersonaEditor(false);
            };

            const resetPrompt = (ab) => {
                if (ab === 'A') { setPersonaPromptA(DEFAULT_PROMPT_A); localStorage.setItem('personaPromptA', DEFAULT_PROMPT_A); }
                else { setPersonaPromptB(DEFAULT_PROMPT_B); localStorage.setItem('personaPromptB', DEFAULT_PROMPT_B); }
            };

            // ── Web search toggle ──
            // Default ON for fresh profiles (key absent); explicit 'false' stays OFF.
            const [webSearch, setWebSearch] = useState(() => localStorage.getItem('webSearch') !== 'false');
            // Defined here (after webSearch) so the ternary can read it correctly
            const THINKING_MSGS = webSearch
                ? ['Thinking...', 'Searching the web...', 'Processing...', 'On it...']
                : ['Thinking...', 'Processing...', 'Analyzing...', 'On it...'];
            const toggleWebSearch = () => {
                const next = !webSearch;
                setWebSearch(next);
                localStorage.setItem('webSearch', String(next));
            };
            // Sync webSearch when changed from Settings
            useEffect(() => {
                const handler = () => setWebSearch(localStorage.getItem('webSearch') !== 'false');
                window.addEventListener('websearch-changed', handler);
                return () => window.removeEventListener('websearch-changed', handler);
            }, []);

            // ── Prompt dropdown ──
            const [showPromptDropdown, setShowPromptDropdown] = useState(false);
            const promptDropdownRef = React.useRef(null);
            React.useEffect(() => {
                const handler = (e) => {
                    if (promptDropdownRef.current && !promptDropdownRef.current.contains(e.target))
                        setShowPromptDropdown(false);
                };
                document.addEventListener('mousedown', handler);
                return () => document.removeEventListener('mousedown', handler);
            }, []);
            React.useEffect(() => {
                const h = (e) => { if (addMenuRef.current && !addMenuRef.current.contains(e.target)) setShowAddMenu(false); };
                document.addEventListener('mousedown', h);
                return () => document.removeEventListener('mousedown', h);
            }, []);

            // @mention file tagging state
            const [mentionActive, setMentionActive] = useState(false);
            const [mentionQuery, setMentionQuery] = useState('');
            const [mentionIndex, setMentionIndex] = useState(0);
            const [taggedFiles, setTaggedFiles] = useState([]); // [{id, filename}]
            // Folder tags: live session scope (server re-expands per request).
            const [taggedFolders, setTaggedFolders] = useState([]); // [{id, filename}]
            const [chatFolderScopes, setChatFolderScopes] = useState({});
            // Session file scope: chatId -> [file ids]. Seeded from the server
            // (GET /chats projection) so scope survives reload; follow-ups,
            // regenerates and resends re-attach it. [] = explicitly cleared.
            const [chatScopes, setChatScopes] = useState({});
            const scopeClearedRef = useRef(false);
            const pendingScopeRef = useRef(null); // scope to commit once the chat id exists
            const pendingFolderScopeRef = useRef(null); // folder scope, same lifecycle
            const [showScopeManager, setShowScopeManager] = useState(false);
            const [showAddMenu, setShowAddMenu] = useState(false);
            const [showVaultPicker, setShowVaultPicker] = useState(false);
            const [vaultPickerSearch, setVaultPickerSearch] = useState('');
            const [vaultPickerSelected, setVaultPickerSelected] = useState(new Set());
            const addMenuRef = React.useRef(null);
            const [expandedFolderId, setExpandedFolderId] = useState(null);
            const [mentionTab, setMentionTab] = useState('recent');

            // Seed taggedFiles when navigated here from Dashboard with pre-selected files
            React.useEffect(() => {
                if (!preTagIds || preTagIds.length === 0) return;
                const toTag = files
                    .filter(f => preTagIds.includes(f.id))
                    .map(f => ({ id: f.id, filename: f.original_filename || f.filename }));
                if (toTag.length > 0) setTaggedFiles(toTag);
            }, [preTagIds]);

            const [indexingTagIds, setIndexingTagIds] = useState(new Set()); // IDs of chat-uploaded files still being indexed
            const indexingPollRef = useRef(null);
            const chatUploadedIdsRef = useRef(new Set()); // IDs uploaded from chat — skip grant prompt for these
            const pgChatIdRef = useRef(null);
            const mentionPopupRef = useRef(null);

            // Poll until all indexing tagged files are ready
            useEffect(() => {
                if (indexingTagIds.size === 0) {
                    clearInterval(indexingPollRef.current);
                    return;
                }
                clearInterval(indexingPollRef.current);
                indexingPollRef.current = setInterval(async () => {
                    const stillPending = new Set();
                    for (const fid of indexingTagIds) {
                        try {
                            const status = await api.getAIStatus(fid);
                            if (isIndexActive(status)) stillPending.add(fid);
                            else if (!isIndexSearchable(status)) {
                                chatUploadedIdsRef.current.delete(fid);
                                const uploadedFile = files.find(file => file.id === fid);
                                showToast(`${uploadedFile?.filename || `File #${fid}`}: ${getIndexStateLabel(status)}`);
                            }
                        } catch(e) {
                            stillPending.add(fid);
                        }
                    }
                    setIndexingTagIds(stillPending);
                }, 30000);
                return () => clearInterval(indexingPollRef.current);
            }, [indexingTagIds.size]);

            // Build full folder path string e.g. "Projects / SubA"
            const getFolderPath = (folderId, folderList) => {
                if (!folderId || !folderList?.length) return null;
                const map = Object.fromEntries(folderList.map(f => [f.id, f]));
                const parts = [];
                let cur = map[folderId];
                while (cur) { parts.unshift(cur.name); cur = map[cur.parent_id]; }
                return parts.join(' / ');
            };

            // Filter files + folders for @mention popup
            const mentionResults = useMemo(() => {
                if (!mentionActive) return { folders: [], files: [], allFolders: [], folderCount: 0, total: 0, flatItems: [], expandedChildren: [], flatIndexMap: {} };
                const q = mentionQuery.toLowerCase().trim();
                const nonDeletedFiles = files.filter(f => !f.is_deleted).map(f => {
                    const path = getFolderPath(f.folder_id, folders);
                    return { ...f, _type:'file', _path: path, _fullPath: path ? `${path} / ${f.filename}` : f.filename };
                });
                const folderItems = folders.map(f => {
                    const path = getFolderPath(f.parent_id, folders);
                    return {
                        id: `folder:${f.id}`, _type:'folder', filename: f.name,
                        mime_type: null, _folderId: f.id, file_count: f.file_count,
                        _path: path, _fullPath: path ? `${path} / ${f.name}` : f.name
                    };
                });
                const alreadyTaggedIds = new Set(taggedFiles.map(t => t.id));

                // Folders tab: ALL folders (no query filter, no cap)
                const allFolders = folderItems
                    .filter(f => !alreadyTaggedIds.has(f.id))
                    .sort((a, b) => (a._fullPath || '').localeCompare(b._fullPath || ''));

                // Files tab: filtered folders (query only) + files
                const filteredFolderItems = folderItems
                    .filter(f => !alreadyTaggedIds.has(f.id))
                    .filter(f => {
                        if (!q) return true;
                        return (f._fullPath || f.filename || '').toLowerCase().includes(q) ||
                            String(f.id).includes(q);
                    });
                const filteredFileItems = nonDeletedFiles
                    .filter(f => !alreadyTaggedIds.has(f.id))
                    .filter(f => {
                        if (!q) return true;
                        return (f._fullPath || f.filename || '').toLowerCase().includes(q) ||
                            String(f.id).includes(q);
                    });
                // Folders only appear when user has typed a search query
                const showFolders = q.length > 0;
                const folderSlice = showFolders ? filteredFolderItems.slice(0, 5) : [];
                // Files tab: alphabetical A-Z index (Recent owns chronological
                // order, so the two tabs never render the same list).
                const fileSlice = filteredFileItems
                    .sort((a, b) => (a._fullPath || a.filename || '').localeCompare(b._fullPath || b.filename || ''))
                    .slice(0, 10);
                // Recent tab: most recently ADDED files first (uploaded_at
                // desc, created_at fallback), same query filter, capped.
                const recentFileItems = nonDeletedFiles
                    .filter(f => !alreadyTaggedIds.has(f.id))
                    .sort((a, b) => (Date.parse(b.uploaded_at || b.created_at || '') || 0) - (Date.parse(a.uploaded_at || a.created_at || '') || 0))
                    .filter(f => {
                        if (!q) return true;
                        return (f._fullPath || f.filename || '').toLowerCase().includes(q) ||
                            String(f.id).includes(q);
                    })
                    .slice(0, 10);

                // Compute expanded folder's child files
                let expandedChildren = [];
                if (expandedFolderId) {
                    expandedChildren = nonDeletedFiles
                        .filter(f => f.folder_id === expandedFolderId && !alreadyTaggedIds.has(f.id))
                        .slice(0, 20);
                }

                // Build flat list for keyboard navigation based on active tab
                const flatItems = [];
                if (mentionTab === 'folders') {
                    // Folders tab: all folders + expanded children
                    allFolders.forEach(f => {
                        flatItems.push(f);
                        if (f._folderId === expandedFolderId) {
                            expandedChildren.forEach(c => flatItems.push(c));
                        }
                    });
                } else if (mentionTab === 'recent') {
                    // Recent tab: recent uploads only, no folder expansion
                    recentFileItems.forEach(f => flatItems.push(f));
                } else {
                    // Files tab: query-filtered folders + expanded children + files
                    folderSlice.forEach(f => {
                        flatItems.push(f);
                        if (f._folderId === expandedFolderId) {
                            expandedChildren.forEach(c => flatItems.push(c));
                        }
                    });
                    fileSlice.forEach(f => flatItems.push(f));
                }

                return {
                    folders: mentionTab === 'recent' ? [] : folderSlice,
                    files: mentionTab === 'recent' ? recentFileItems : fileSlice,
                    allFolders,
                    folderCount: folderSlice.length,
                    total: flatItems.length,
                    flatItems,
                    expandedChildren,
                    flatIndexMap: Object.fromEntries(flatItems.map((item, i) => [item.id, i])),
                };
            }, [mentionActive, mentionQuery, files, folders, taggedFiles, expandedFolderId, mentionTab]);

            // Handle input change with @mention detection
            const handleInputChange = (e) => {
                const val = e.target.value;
                setInput(val);

                // Find the last @ in the input from current cursor position
                const cursorPos = e.target.selectionStart;
                const textBeforeCursor = val.substring(0, cursorPos);
                const lastAtIndex = textBeforeCursor.lastIndexOf('@');

                if (lastAtIndex !== -1) {
                    const textAfterAt = textBeforeCursor.substring(lastAtIndex + 1);
                    // Only activate if no space before @, or @ is at start
                    const charBeforeAt = lastAtIndex > 0 ? val[lastAtIndex - 1] : ' ';
                    if (charBeforeAt === ' ' || lastAtIndex === 0) {
                        // Don't reactivate if this @ already resolves to a tagged file or folder
                        const isAlreadyTagged = taggedFiles.some(f => {
                            const display = f._tagDisplay || f.filename;
                            return textAfterAt === display ||
                                textAfterAt.startsWith(display + ' ') ||
                                textAfterAt.startsWith(display + '\n') ||
                                textAfterAt.startsWith(display + '?') ||
                                textAfterAt.startsWith(display + '/');
                        }) || taggedFolders.some(f => {
                            const candidates = [f.filename, f._fullPath].filter(Boolean);
                            return candidates.some(display =>
                                textAfterAt === display ||
                                textAfterAt === display + '/' ||
                                textAfterAt.startsWith(display + ' ') ||
                                textAfterAt.startsWith(display + '\n') ||
                                textAfterAt.startsWith(display + '?') ||
                                textAfterAt.startsWith(display + '/') ||
                                textAfterAt.startsWith(display.replace(/ \/ /g, '/') + '/')
                            );
                        });
                        if (!isAlreadyTagged) {
                            setMentionActive(true);
                            setMentionQuery(textAfterAt);
                            setMentionIndex(0);
                            return;
                        }
                    }
                }
                setMentionActive(false);
                setMentionQuery('');
            };

            // Tag a whole folder as live session scope (server re-expands
            // membership per request, so later uploads join automatically).
            const selectMentionFolder = (item) => {
                const folderId = item._folderId;
                const folderName = item.filename;
                setTaggedFolders(prev => prev.some(f => f.id === folderId)
                    ? prev
                    : [...prev, { id: folderId, filename: folderName }]);
                const cursorPos = inputRef.current?.selectionStart || input.length;
                const textBeforeCursor = input.substring(0, cursorPos);
                const lastAtIndex = textBeforeCursor.lastIndexOf('@');
                const textAfter = input.substring(cursorPos);
                const replacement = `@${item._fullPath || folderName}/ `;
                const newInput = input.substring(0, lastAtIndex) + replacement + textAfter;
                setInput(newInput);
                setMentionActive(false);
                setMentionQuery('');
                setMentionIndex(0);
                setTimeout(() => inputRef.current?.focus(), 50);
            };

            const removeTaggedFolder = (folderId) => {
                setTaggedFolders(prev => prev.filter(f => f.id !== folderId));
                setTimeout(() => inputRef.current?.focus(), 50);
            };

            // Select a file from the @mention popup
            const selectMention = (item) => {
                // If it's a folder, toggle expansion to show child files
                if (item._type === 'folder') {
                    setExpandedFolderId(prev => prev === item._folderId ? null : item._folderId);
                    setMentionIndex(0);
                    return;
                }

                // Regular file (or expanded child file). Unsupported types are
                // blocked here with a reason — attaching them only manufactures
                // a failure card two steps later. Unindexed-but-supported files
                // attach with a warning flag; the grant flow still runs at send.
                const verdict = classifyTagTarget(item);
                if (verdict.verdict === 'unsupported') {
                    showToast(verdict.reason, 'error', null, 5000);
                    return;
                }
                setTaggedFiles(prev => [...prev, { id: item.id, filename: item.filename, _tagDisplay: item._fullPath || item.filename, ...(verdict.verdict === 'unindexed' ? { _warn: verdict.reason } : {}) }]);

                // Replace @query with @filename in the input text
                const cursorPos = inputRef.current?.selectionStart || input.length;
                const textBeforeCursor = input.substring(0, cursorPos);
                const lastAtIndex = textBeforeCursor.lastIndexOf('@');
                const textAfter = input.substring(cursorPos);

                const replacement = `@${item._fullPath || item.filename} `;
                const newInput = input.substring(0, lastAtIndex) + replacement + textAfter;
                const newCursorPos = lastAtIndex + replacement.length;

                setInput(newInput);

                setMentionActive(false);
                setMentionQuery('');
                setMentionIndex(0);

                // Re-focus and set cursor position after the inserted name
                setTimeout(() => {
                    if (inputRef.current) {
                        inputRef.current.focus();
                        inputRef.current.setSelectionRange(newCursorPos, newCursorPos);
                    }
                }, 50);
            };

            const tagAllInFolder = (folderId) => {
                const folderFiles = files.filter(f => f.folder_id === folderId && !f.is_deleted);
                if (!folderFiles.length) return;
                const blocked = folderFiles.filter(f => classifyTagTarget(f).verdict === 'unsupported');
                if (blocked.length > 0) {
                    showToast(`Skipped ${blocked.length} unreadable file${blocked.length === 1 ? '' : 's'} (zip/video can't be indexed).`, 'error', null, 5000);
                }
                setTaggedFiles(prev => {
                    const existingIds = new Set(prev.map(t => t.id));
                    const newOnes = folderFiles
                        .filter(f => !existingIds.has(f.id) && classifyTagTarget(f).verdict !== 'unsupported')
                        .map(f => {
                            const verdict = classifyTagTarget(f);
                            return {
                                id: f.id, filename: f.filename, _tagDisplay: f.filename,
                                ...(verdict.verdict === 'unindexed' ? { _warn: verdict.reason } : {}),
                            };
                        });
                    return [...prev, ...newOnes];
                });
                setExpandedFolderId(null);
                setMentionActive(false);
                setMentionQuery('');
                setTimeout(() => inputRef.current?.focus(), 50);
            };

            // Remove a tagged file
            const removeTaggedFile = (fileId) => {
                setTaggedFiles(prev => prev.filter(f => f.id !== fileId));
                setTimeout(() => inputRef.current?.focus(), 50);
            };

            // Composer image previews: blob URLs for tagged images, cached
            // per file id, revoked when untagged or on unmount. Failures
            // fall back to the filename pill silently.
            const isImageFilename = (name) => isImageFile({ filename: name });
            const [thumbUrls, setThumbUrls] = useState({});
            const thumbCacheRef = useRef({});
            const [thumbFailed, setThumbFailed] = useState({});
            useEffect(() => {
                let cancelled = false;
                const live = new Set(taggedFiles.map(f => f.id));
                // Previews appear only once indexed: indexing images show
                // a shimmer placeholder meanwhile (never a filename pill).
                const indexed = new Set([...live].filter(id => !indexingTagIds.has(id)));
                for (const id of Object.keys(thumbCacheRef.current)) {
                    if (!indexed.has(Number(id)) && !indexed.has(id)) {
                        api.revokeFileObjectUrl(thumbCacheRef.current[id]);
                        delete thumbCacheRef.current[id];
                    }
                }
                setThumbUrls(prev => {
                    const next = {};
                    for (const id of Object.keys(prev)) {
                        if (thumbCacheRef.current[id]) next[id] = thumbCacheRef.current[id];
                    }
                    return next;
                });
                setThumbFailed(prev => {
                    const next = {};
                    for (const id of Object.keys(prev)) {
                        if (live.has(Number(id)) || live.has(id)) next[id] = true;
                    }
                    return next;
                });
                const want = taggedFiles.filter(tf => isImageFilename(tf.filename) && indexed.has(tf.id) && !thumbCacheRef.current[tf.id]);
                if (!want.length) return;
                (async () => {
                    for (const tf of want) {
                        try {
                            const url = await api.fetchFileObjectUrl(tf.id);
                            if (cancelled) { api.revokeFileObjectUrl(url); continue; }
                            thumbCacheRef.current[tf.id] = url;
                            if (!cancelled) setThumbUrls(prev => ({ ...prev, [tf.id]: url }));
                        } catch (e) {
                            if (!cancelled) setThumbFailed(prev => ({ ...prev, [tf.id]: true }));
                        }
                    }
                })();
                return () => { cancelled = true; };
            }, [taggedFiles, indexingTagIds]);
            useEffect(() => {
                const cache = thumbCacheRef.current;
                return () => {
                    for (const id of Object.keys(cache)) api.revokeFileObjectUrl(cache[id]);
                };
            }, []);
            // Multiline composer: grow with content up to the CSS cap,
            // then scroll. Runs on every input change (typing, send
            // clearing, mention inserts) — cheap DOM measurement only.
            useEffect(() => {
                const el = inputRef.current;
                if (!el || el.tagName !== 'TEXTAREA') return;
                el.style.height = 'auto';
                const max = 160;
                el.style.height = Math.min(el.scrollHeight, max) + 'px';
                el.style.overflowY = el.scrollHeight > max ? 'auto' : 'hidden';
            }, [input]);

            // Scope-manager edits: optimistic local update + immediate server
            // PATCH (server is authority; reload reseeds from it). Failure
            // snaps back to the pre-edit set with a toast.
            const removeScopeFile = async (fileId) => {
                const key = pgChatId || pgChatIdRef.current;
                if (!key) return;
                const prev = normalizeScopeIds(chatScopes[key]);
                const next = prev.filter(id => id !== fileId);
                setChatScopes(p => ({ ...p, [key]: next }));
                try {
                    await api.setChatScope(key, next);
                } catch (e) {
                    setChatScopes(p => ({ ...p, [key]: prev }));
                    showToast('Could not update scope — restored previous files.', 'error');
                }
                if (next.length === 0) setShowScopeManager(false);
            };

            // Esc closes the scope manager without changing anything.
            React.useEffect(() => {
                if (!showScopeManager) return undefined;
                const handler = (e) => { if (e.key === 'Escape') setShowScopeManager(false); };
                document.addEventListener('keydown', handler);
                return () => document.removeEventListener('keydown', handler);
            }, [showScopeManager]);

            const removeScopeFolder = async (folderId) => {
                const key = pgChatId || pgChatIdRef.current;
                if (!key) return;
                const prevFiles = normalizeScopeIds(chatScopes[key]);
                const prevFolders = normalizeScopeIds(chatFolderScopes[key]);
                const next = prevFolders.filter(id => id !== folderId);
                setChatFolderScopes(p => ({ ...p, [key]: next }));
                try {
                    await api.setChatScope(key, prevFiles, next);
                } catch (e) {
                    setChatFolderScopes(p => ({ ...p, [key]: prevFolders }));
                    showToast('Could not update scope — restored previous folders.', 'error');
                }
                if (next.length === 0 && prevFiles.length === 0) setShowScopeManager(false);
            };

            const clearScopeNow = async () => {                const key = pgChatId || pgChatIdRef.current;
                if (key) {
                    const prev = normalizeScopeIds(chatScopes[key]);
                    const prevFolders = normalizeScopeIds(chatFolderScopes[key]);
                    setChatScopes(p => ({ ...p, [key]: [] }));
                    setChatFolderScopes(p => ({ ...p, [key]: [] }));
                    try {
                        await api.setChatScope(key, [], []);
                    } catch (e) {
                        setChatScopes(p => ({ ...p, [key]: prev }));
                        setChatFolderScopes(p => ({ ...p, [key]: prevFolders }));
                        showToast('Could not clear scope.', 'error');
                        return;
                    }
                }
                scopeClearedRef.current = true;
                setShowScopeManager(false);
            };

            // Folders-only clear for the folder pill ×: composer drafts and
            // session folders go, file scope is never touched.
            const clearFolderScopeNow = async () => {
                const key = pgChatId || pgChatIdRef.current;
                const prevDrafts = [...taggedFolders];
                setTaggedFolders([]);
                if (key) {
                    const prevFolders = normalizeScopeIds(chatFolderScopes[key]);
                    setChatFolderScopes(p => ({ ...p, [key]: [] }));
                    try {
                        await api.setChatScope(key, null, []);
                    } catch (e) {
                        setTaggedFolders(prevDrafts);
                        setChatFolderScopes(p => ({ ...p, [key]: prevFolders }));
                        showToast('Could not clear folder scope.', 'error');
                        return;
                    }
                }
                setShowScopeManager(false);
            };

            // Handle keyboard navigation in @mention popup
            // Pasted chat images: upload into the auto-provisioned
            // chat_uploads folder and tag them exactly like picker
            // uploads. Text in the same paste still lands in the input.
            const handleChatPaste = async (e) => {
                const blobs = [];
                try {
                    const items = Array.from(e.clipboardData?.items || []);
                    for (const it of items) {
                        if (it.kind === 'file' && it.type?.startsWith('image/')) {
                            const f = it.getAsFile();
                            if (f) blobs.push(f);
                        }
                    }
                    if (!blobs.length && e.clipboardData?.files) {
                        for (const f of Array.from(e.clipboardData.files)) {
                            if (f.type?.startsWith('image/')) blobs.push(f);
                        }
                    }
                } catch (err) { return; }
                if (!blobs.length) return;
                // Timestamp without regex character classes: Tailwind scans
                // this file for class candidates and chokes on [...] tokens.
                const now = new Date();
                const two = (n) => String(n).padStart(2, '0');
                const stamp = `${now.getFullYear()}${two(now.getMonth() + 1)}${two(now.getDate())}${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}`;
                let n = 0;
                for (const blob of blobs) {
                    const rawExt = ((blob.type || '').split('/')[1] || 'png').toLowerCase();
                    let ext = '';
                    for (const ch of rawExt) {
                        if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) ext += ch;
                    }
                    if (!ext) ext = 'png';
                    const name = `pasted-image-${stamp}${n ? '-' + n : ''}.${ext}`;
                    n += 1;
                    const file = new File([blob], name, { type: blob.type || 'image/png' });
                    // Marker: replacements of this file stay in chat_uploads
                    // (handleChatReplace forwards it as the folder target).
                    file.chatUploadFolder = 'chat_uploads';
                    try {
                        const uploaded = await api.uploadFile(file, false, null, null, 'chat_uploads');
                        const fid = uploaded.file_id;
                        setTaggedFiles(prev => [...prev, { id: fid, filename: uploaded.filename }]);
                        try {
                            await api.grantAIAccess(fid);
                            chatUploadedIdsRef.current.add(fid);
                            setIndexingTagIds(prev => new Set([...prev, fid]));
                        } catch (grantError) {
                            showToast(describeIndexingError(grantError, file.name), 'info');
                        }
                        refresh && refresh();
                    } catch (err) {
                        const isDuplicate = err.data && err.data.existing_id;
                        if (isDuplicate) {
                            const eid = err.data.existing_id;
                            setTaggedFiles(prev => {
                                const alreadyTagged = prev.some(t => t.id === eid);
                                if (alreadyTagged) return prev;
                                return [...prev, { id: eid, filename: err.data.filename }];
                            });
                            showToast(`"${file.name}" already exists — tagged existing file.`, 'error', file);
                        } else {
                            showToast(`${file.name}: ${err.message}`, 'error', file);
                        }
                    }
                }
            };
            const handleInputKeyDown = (e) => {
                if (mentionActive && mentionResults.total > 0) {
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        setMentionIndex(prev => Math.min(prev + 1, mentionResults.total - 1));
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        setMentionIndex(prev => Math.max(prev - 1, 0));
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const item = mentionResults.flatItems[mentionIndex];
                        if (item) selectMention(item);
                        return;
                    }
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        setMentionTab(prev => prev === 'recent' ? 'files' : prev === 'files' ? 'folders' : 'recent');
                        setMentionIndex(0);
                        return;
                    }
                    if (e.key === 'Escape') {
                        e.preventDefault();
                        setMentionActive(false);
                        return;
                    }
                }
                // Enter sends when the picker is closed OR open-but-empty
                // (an empty picker previously swallowed Enter entirely).
                // With results present, Enter still selects the item above.
                // preventDefault only on send: a textarea would otherwise
                // insert a newline alongside sending.
                if (e.key === 'Enter' && !e.shiftKey && prefSendOnEnter !== false && (!mentionActive || mentionResults.total === 0)) {
                    e.preventDefault();
                    sendWithTags();
                }
            };

            // Scroll focused mention item into view
            useEffect(() => {
                if (!mentionActive || mentionResults.total === 0) return;
                const container = mentionPopupRef.current?.querySelector('[class*="overflow-y-auto"]') || mentionPopupRef.current;
                if (!container) return;
                const focused = container.querySelector(`[data-mention-idx="${mentionIndex}"]`);
                if (focused) focused.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }, [mentionIndex, mentionActive, mentionResults.total]);

            // Reset expanded folder when menu closes or query changes
            useEffect(() => {
                if (!mentionActive) {
                    setExpandedFolderId(null);
                    setMentionTab('recent');
                }
            }, [mentionActive]);

            const GREETINGS = [
                "Hey! What can I help you find today?",
                "Ready when you are. What's on your mind?",
                "Your vault is open. What would you like to explore?",
                "Hello! Ask me anything about your files.",
                "What are we diving into today?",
                "All set. Drop a question or tag a file to get started.",
                "Good to see you. What do you need?",
                "I'm here. What would you like to know?",
                "Your files are ready. What's the question?",
                "Let's get into it — what are you looking for?",
            ];

            const normalizeMsg = (m) => ({
                ...m,
                content: m.role === 'assistant' ? stripAssistantProtocolArtifacts(m.content || '') : (m.content || ''),
                sources: m.sources || [],
                followups: normalizeFollowupSuggestions(m.followups),
                streaming: false,
                created_at: m.created_at || new Date().toISOString(),
            });

            useEffect(() => {
                const generation = ++historyLoadGenerationRef.current;
                const isCurrent = () => historyLoadGenerationRef.current === generation;
                const skippedSelection = skipHistoryLoadRef.current;
                if (
                    skippedSelection
                    && skippedSelection.sessionId === sessionId
                    && skippedSelection.pgChatId === pgChatId
                ) {
                    skipHistoryLoadRef.current = null;
                    setLoadingHistory(false);
                    return () => {
                        if (isCurrent()) historyLoadGenerationRef.current += 1;
                    };
                }
                skipHistoryLoadRef.current = null;
                hasRunDiscoveryRef.current = false;
                const loadHistory = async () => {
                    if (processing) { if (isCurrent()) setLoadingHistory(false); return; } // don't overwrite mid-stream
                    setLoadingHistory(true);
                    try {
                        if (pgChatId) {
                            // Load messages from PostgreSQL persistent chat
                            const data = await api.getChat(pgChatId);
                            if (!isCurrent()) return;
                            const msgs = Array.isArray(data.messages) ? data.messages : [];
                            if (msgs.length > 0) {
                                setMessages(msgs.map(normalizeMsg));
                            } else {
                                const greeting = GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
                                setMessages([{ role: 'assistant', content: greeting, created_at: new Date().toISOString() }]);
                            }
                            // Seed session file scope so follow-ups stay scoped after reload.
                            const seed = normalizeScopeIds(data.scoped_file_ids);
                            if (seed.length > 0) {
                                setChatScopes(prev => ({ ...prev, [pgChatId]: seed }));
                            }
                            const folderSeed = normalizeScopeIds(data.scoped_folder_ids);
                            if (folderSeed.length > 0) {
                                setChatFolderScopes(prev => ({ ...prev, [pgChatId]: folderSeed }));
                            }
                        } else {
                            const data = await api.getChatHistory(sessionId);
                            if (!isCurrent()) return;
                            if (data.history && data.history.length > 0) {
                                setMessages(data.history.map(normalizeMsg));
                            } else {
                                const greeting = GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
                                setMessages([{ role: 'assistant', content: greeting, created_at: new Date().toISOString() }]);
                            }
                        }
                    } catch (e) {
                        if (!isCurrent()) return;
                        console.error('History load failed:', e);
                        const greeting = GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
                        setMessages([{ role: 'assistant', content: greeting, created_at: new Date().toISOString() }]);
                    } finally {
                        if (isCurrent()) setLoadingHistory(false);
                    }
                };
                loadHistory();
                return () => {
                    if (isCurrent()) historyLoadGenerationRef.current += 1;
                };
            }, [sessionId, pgChatId, processing]);

            useEffect(() => {
                if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }, [messages, grantPromptFiles]);

            const toggleFile = (msgIndex, fileId) => {
                setSelectedIds(prev => {
                    const current = prev[msgIndex] || [];
                    const next = current.includes(fileId)
                        ? current.filter(id => id !== fileId)
                        : [...current, fileId];
                    return { ...prev, [msgIndex]: next };
                });
            };

            const [resolvedMsgs, setResolvedMsgs] = useState({}); // Track resolved file suggestion messages

            const showFileSuggestions = (text, candidates, totalFiles) => {
                const matchedFiles = (Array.isArray(candidates) ? candidates : [])
                    .map(file => ({
                        ...file,
                        id: file.id ?? file.file_id,
                        match_percentage: resolveMatchPercentage(file),
                    }))
                    .filter(file => file.file_id && file.match_percentage > 0)
                    .slice(0, 5);
                setMessages(prev => {
                    const msgIndex = prev.length;
                    setSelectedIds(selected => ({
                        ...selected,
                        [msgIndex]: matchedFiles.map(file => file.file_id),
                    }));
                    return [...prev, {
                        role: 'system',
                        content: matchedFiles.length > 0
                            ? `We have ${totalFiles ?? '?'} files in vault. Found ${matchedFiles.length} relevant. Choose the files to process.`
                            : `I couldn't find a relevant indexed file. You can continue without vault context.`,
                        matchedFiles,
                        msgIndex,
                        originalInput: text,
                    }];
                });
            };

            // Wrapper to send message with tagged files
            const sendWithTags = () => {
                if (!input.trim() || processing) return;
                if (indexingTagIds.size > 0) return; // wait for vector indexing
                const tagIds = taggedFiles.map(f => f.id);
                const tagNames = taggedFiles.map(f => f.filename);
                let textToSend = input.trim();
                // Drop a trailing unresolved @query (picker was open): scope
                // already travels via tagIds/tagFolderIds state, so the raw
                // tail is only noise for the model. Anchored on the picker's
                // own query text — plain "@" mentions elsewhere are untouched.
                if (mentionActive && mentionQuery) {
                    const tail = '@' + mentionQuery;
                    if (textToSend.endsWith(tail)) {
                        textToSend = textToSend.substring(0, textToSend.length - tail.length).trim();
                    }
                }

                // Check if any tagged file is not AI-indexed — always prompt for permission
                if (taggedFiles.length > 0) {
                    const unindexed = taggedFiles.filter(tf => {
                        // Skip grant prompt for files uploaded from chat — access already granted
                        if (chatUploadedIdsRef.current.has(tf.id)) return false;
                        const fullFile = files.find(f => f.id === tf.id);
                        if (!fullFile) return false;
                        return !isIndexSearchable(fullFile);
                    });
                    if (unindexed.length > 0) {
                        setMessages(prev => [...prev, {
                            role: 'user',
                            content: textToSend,
                            taggedFiles: tagNames.length > 0 ? tagNames : null,
                            created_at: new Date().toISOString(),
                        }]);
                        setInput('');
                        setTaggedFiles([]);
                        setIndexingTagIds(new Set());
                        setMentionActive(false);
                        setGrantPromptFiles(unindexed);
                        setPendingSend({ text: textToSend, tagIds, tagNames, tagFolderIds: taggedFolders.map(f => f.id) });
                        return;
                    }
                }

                // Set user message with tagged files stored separately for styled rendering
                setMessages(prev => [...prev, {
                    role: 'user',
                    content: textToSend,
                    taggedFiles: tagNames.length > 0 ? tagNames : null,
                    created_at: new Date().toISOString(),
                }]);

                setInput('');
                setTaggedFiles([]);
                setIndexingTagIds(new Set());
                // Capture chat-uploaded IDs before clearing — passed to backend for direct-read path
                const capturedChatUploadIds = Array.from(chatUploadedIdsRef.current);
                chatUploadedIdsRef.current = new Set();
                setMentionActive(false);

                // Additive scoping: fresh @-tags union into the live session
                // scope instead of replacing it. Clearing stays explicit.
                // Folders union the same way (live server-side expansion).
                const scopeKey = pgChatId || pgChatIdRef.current || null;
                const liveScope = normalizeScopeIds(scopeKey ? chatScopes[scopeKey] : []);
                const liveFolders = normalizeScopeIds(scopeKey ? chatFolderScopes[scopeKey] : []);
                const mergedTags = unionScopeIds(tagIds, liveScope);
                const mergedFolders = unionScopeIds(taggedFolders.map(f => f.id), liveFolders);
                const sendFileIds = mergedTags.length > 0
                    ? mergedTags
                    : (scopeClearedRef.current ? [] : null);
                const sendFolderIds = mergedFolders.length > 0
                    ? mergedFolders
                    : (scopeClearedRef.current ? [] : null);
                scopeClearedRef.current = false;
                // Mirror what was sent: ids (or []) become the chat scope once
                // the chat id exists; null changes nothing server-side either.
                pendingScopeRef.current = sendFileIds === null ? null : [...sendFileIds];
                pendingFolderScopeRef.current = sendFolderIds === null ? null : [...sendFolderIds];
                setTaggedFolders([]);
                sendRaw(textToSend, null, sendFileIds, null, true, capturedChatUploadIds.length > 0 ? capturedChatUploadIds : null, false, sendFolderIds);
            };

            // After granting access, continue with pending send
            const handleGrantAndSend = async () => {
                if (!grantPromptFiles || !pendingSend) return;
                const filesToIndex = grantPromptFiles;
                const { text, tagIds, tagNames } = pendingSend;
                const pendingTagFolders = pendingSend.tagFolderIds || [];
                // Grant access for each unindexed file
                const grantedIds = [];
                for (const f of filesToIndex) {
                    try {
                        onGrantAccess && await onGrantAccess(f.id);
                        grantedIds.push(f.id);
                    } catch (e) {
                        showToast(describeIndexingError(e, f.filename || `File #${f.id}`), 'info');
                    }
                }
                setGrantPromptFiles(null);
                setPendingSend(null);
                if (!grantedIds.length) {
                    setInput(text);
                    return;
                }
                // Hold the send until fresh grants finish indexing instead of
                // firing into the backend's instant fail-fast.
                setGrantDone(true);
                const waitDeadline = Date.now() + 35_000;
                let allReady = false;
                while (Date.now() < waitDeadline) {
                    const states = await Promise.all(
                        grantedIds.map(id => api.getAIStatus(id).catch(() => ({ state: 'queued' })))
                    );
                    // Recomputed fresh every poll — a single weird response can
                    // never latch a stale "still indexing" decision.
                    allReady = states.every(status => !isIndexActive(status));
                    if (allReady) break;
                    await new Promise(r => setTimeout(r, 1500));
                }
                setGrantDone(false);
                if (!allReady) {
                    showToast('Still indexing — your message is kept. Send again in a moment.', 'info');
                    setInput(text);
                    return;
                }
                const grantFileIds = tagIds.length > 0
                    ? unionScopeIds(tagIds, normalizeScopeIds(chatScopes[pgChatId || pgChatIdRef.current] || []))
                    : null;
                const grantFolderIds = unionScopeIds(
                    pendingTagFolders,
                    normalizeScopeIds(chatFolderScopes[pgChatId || pgChatIdRef.current] || []),
                );
                pendingScopeRef.current = grantFileIds === null ? null : [...grantFileIds];
                pendingFolderScopeRef.current = grantFolderIds.length > 0 ? [...grantFolderIds] : null;
                sendRaw(text, null, grantFileIds, null, true, null, false, grantFolderIds.length > 0 ? grantFolderIds : null);
            };



            const send = async (overrideInput = null, confirm = null, fileIds = null, msgIndexToResolve = null, folderIds = null) => {
                return sendRaw(overrideInput || input, confirm, fileIds, msgIndexToResolve, msgIndexToResolve !== null, null, false, folderIds);
            };

            const sessionFolderIds = () => followupFileIds(chatFolderScopes[pgChatId || pgChatIdRef.current] || []);

            const cancelRequest = () => {
                if (abortRef.current) {
                    abortRef.current.abort();
                    abortRef.current = null;
                }
                setProcessing(false);
            };

            const copyMessage = async (content, btn) => {
                try {
                    if (navigator.clipboard?.writeText) {
                        await navigator.clipboard.writeText(content);
                    } else {
                        throw new Error('no clipboard API');
                    }
                } catch(e) {
                    const ta = document.createElement('textarea');
                    ta.value = content;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                }
                if (btn) { const orig = btn.textContent; btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = orig, 2000); }
            };

            const regenerateMessage = (idx) => {
                if (processing) return;
                // Find the user message before this assistant message
                let userMsg = null;
                for (let i = idx - 1; i >= 0; i--) {
                    if (messages[i].role === 'user') {
                        userMsg = messages[i].content;
                        break;
                    }
                }
                if (!userMsg) return;
                // Remove the current assistant message and re-send
                setMessages(prev => prev.slice(0, idx));
                // Re-attach session scope: regenerating must not widen to global.
                const regenScope = followupFileIds(chatScopes[pgChatId || pgChatIdRef.current] || []);
                sendRaw(userMsg, null, regenScope, null, true, null, false, sessionFolderIds());
            };

            const sendFollowup = (question) => {
                if (processing) return;
                setInput('');
                setMessages(prev => [...prev, { role: 'user', content: question, created_at: new Date().toISOString() }]);
                // Re-attach session scope; null lets the server session
                // fallback apply (covers reload races).
                const scopeIds = followupFileIds(chatScopes[pgChatId || pgChatIdRef.current] || []);
                sendRaw(question, null, scopeIds, null, true, null, false, sessionFolderIds());
            };

            // Floating selection menu
            const [floatingMenu, setFloatingMenu] = React.useState(null);

            React.useEffect(() => {
                const handleSelection = () => {
                    const selection = window.getSelection();
                    const text = selection?.toString().trim();
                    if (!text || text.length < 3) { setFloatingMenu(null); return; }
                    try {
                        const range = selection.getRangeAt(0);
                        const rect = range.getBoundingClientRect();
                        setFloatingMenu({
                            x: rect.left + rect.width / 2,
                            y: rect.top - 44,
                            text
                        });
                    } catch(e) {}
                };
                const handleClear = () => {
                    if (!window.getSelection()?.toString().trim()) setFloatingMenu(null);
                };
                document.addEventListener('mouseup', handleSelection);
                document.addEventListener('selectionchange', handleClear);
                return () => {
                    document.removeEventListener('mouseup', handleSelection);
                    document.removeEventListener('selectionchange', handleClear);
                };
            }, []);

            // Copy buttons for code blocks
            React.useEffect(() => {
                document.querySelectorAll('.prose pre, .markdown-content pre').forEach(pre => {
                    if (pre.querySelector('.copy-btn')) return;
                    const btn = document.createElement('button');
                    btn.className = 'copy-btn';
                    btn.textContent = 'Copy';
                    btn.onclick = async () => {
                        const text = pre.querySelector('code')?.textContent || pre.textContent;
                        try {
                            if (navigator.clipboard?.writeText) {
                                await navigator.clipboard.writeText(text);
                            } else {
                                throw new Error('no clipboard API');
                            }
                        } catch(e) {
                            const ta = document.createElement('textarea');
                            ta.value = text;
                            ta.style.position = 'fixed';
                            ta.style.opacity = '0';
                            document.body.appendChild(ta);
                            ta.select();
                            document.execCommand('copy');
                            document.body.removeChild(ta);
                        }
                        btn.textContent = 'Copied!';
                        setTimeout(() => btn.textContent = 'Copy', 2000);
                    };
                    pre.style.position = 'relative';
                    pre.appendChild(btn);
                });
            }, [messages]);

            const STATUS_LABELS = {
                'queued':          'Queued...',
                'searching_vault': 'Searching vault...',
                'reranking':       'Reading documents...',
                'generating':      'Generating response...',
                'web_search':      'Searching web...',
                'verifying_citations': 'Verifying sources...',
            };

            const sendRaw = async (text, confirm = null, fileIds = null, msgIndexToResolve = null, skipUserMsg = false, chatUploadIds = null, forceDeepSearch = false, folderIds = null) => {
                if (!text.trim() || processing) return;

                // A history request may have started just before the user sent
                // the first message. It must never replace the user/stream
                // rows after the request has become interactive.
                historyLoadGenerationRef.current += 1;
                setLoadingHistory(false);
                const selectionAtSend = { sessionId, pgChatId };

                if (!skipUserMsg) setInput('');

                let userMessageAlreadyAdded = skipUserMsg;
                let effectiveFileIds = fileIds;
                const shouldDiscoverFiles = !hasRunDiscoveryRef.current
                    && confirm === null
                    && fileSearch
                    && (!fileIds || fileIds.length === 0)
                    && (!chatUploadIds || chatUploadIds.length === 0)
                    && msgIndexToResolve === null;

                if (shouldDiscoverFiles) {
                    hasRunDiscoveryRef.current = true;
                    if (!userMessageAlreadyAdded) {
                        setMessages(prev => [...prev, { role: 'user', content: text, created_at: new Date().toISOString() }]);
                        userMessageAlreadyAdded = true;
                    }
                    setProcessing(true);
                    try {
                        const discovery = await api.searchVault(text, 5);
                        const meaningful = (discovery.results || []).filter(f => (f.match_percentage || 0) >= 35);
                        if (meaningful.length > 0) {
                            // Check if all meaningful files are already indexed
                            const allIndexed = meaningful.every(f => {
                                const fullFile = files.find(ff => ff.id === f.file_id);
                                return !fullFile || isIndexSearchable(fullFile);
                            });
                            if (allIndexed) {
                                // Auto-select all indexed files — skip the prompt.
                                // Additive: union into live session scope.
                                effectiveFileIds = unionScopeIds(
                                    meaningful.map(f => f.file_id),
                                    normalizeScopeIds(chatScopes[pgChatId || pgChatIdRef.current] || []),
                                );
                                const autoTagged = meaningful.map(f => ({
                                    id: f.file_id,
                                    filename: f.filename,
                                    _tagDisplay: f.filename,
                                }));
                                setTaggedFiles(autoTagged);
                                pendingScopeRef.current = [...effectiveFileIds];
                                // Continue to chat send below (don't return)
                            } else {
                                // Some files need permission — show the prompt
                                showFileSuggestions(text, meaningful, discovery.total_files);
                                return;
                            }
                        }
                    } catch (error) {
                        console.warn('Suggested-file discovery unavailable; continuing with agent retrieval.', error);
                    } finally {
                        const currentSelection = activeChatSelectionRef.current;
                        if (
                            currentSelection.sessionId === selectionAtSend.sessionId
                            && currentSelection.pgChatId === selectionAtSend.pgChatId
                        ) {
                            skipHistoryLoadRef.current = selectionAtSend;
                        }
                        setProcessing(false);
                    }
                }

                // Mark the message as resolved if applicable
                if (msgIndexToResolve !== null) {
                    setResolvedMsgs(prev => ({ ...prev, [msgIndexToResolve]: true }));
                }

                if (!confirm && !userMessageAlreadyAdded) {
                    setMessages(prev => [...prev, { role: 'user', content: text, created_at: new Date().toISOString() }]);
                }

                setProcessing(true);
                const controller = new AbortController();
                abortRef.current = controller;

                // Add streaming placeholder message
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: '',
                    streaming: true,
                    statusStep: null,
                    statusDetail: null,
                    statusHistory: [],
                    sources: [],
                    followups: [],
                    usage: null,
                    created_at: new Date().toISOString(),
                }]);

                let createdPgChat = null;
                let requestCompleted = false;
                try {
                    const activePromptText = activePrompt === 'B' ? personaPromptB : personaPromptA;

                    // Auto-create a PG chat if none exists so messages are persisted to PostgreSQL.
                    // When the user clicked "New Chat", pgChatId prop is null and pgChatIdRef
                    // may still hold a stale ID from the previous chat — always force create.
                    let activeChatId = pgChatId || pgChatIdRef.current;
                    if (!activeChatId) {
                        try {
                            const newChat = await api.createChat(text.slice(0, 60) || 'New Chat');
                            activeChatId = newChat.id;
                            createdPgChat = newChat;
                            pgChatIdRef.current = newChat.id;
                        } catch(e) {
                            showToast('Could not create chat session. Check connection and try again.', 'error');
                            setMessages(prev => {
                                const last = prev[prev.length - 1];
                                const cleaned = last && last.role === 'assistant' && last.streaming && !last.content
                                    ? prev.slice(0, -1)
                                    : prev;
                                return [...cleaned, { role: 'system', content: e.message || 'Failed to create chat session.' }];
                    });
                            setProcessing(false);
                            return;
                        }
                    }
                    // Commit any pending session scope now that the chat id is
                    // known (covers brand-new chats created above).
                    if (pendingScopeRef.current !== null && activeChatId) {
                        const committed = pendingScopeRef.current;
                        pendingScopeRef.current = null;
                        setChatScopes(prev => ({ ...prev, [activeChatId]: committed }));
                    }
                    if (pendingFolderScopeRef.current !== null && activeChatId) {
                        const committedFolders = pendingFolderScopeRef.current;
                        pendingFolderScopeRef.current = null;
                        setChatFolderScopes(prev => ({ ...prev, [activeChatId]: committedFolders }));
                    }

                    const res = await api.chat({
                        message: text,
                        fileIds: effectiveFileIds,
                        folderIds: folderIds !== undefined ? folderIds : null,
                        mode: activePrompt === 'B' ? 'expert' : 'casual',
                        deepSearch: forceDeepSearch || fileSearch,
                        webSearch,
                        personaPrompt: activePromptText,
                        signal: controller.signal,
                        // Append each server delta immediately. The transport is the pacing
                        // source; the UI never manufactures a delayed typewriter animation.
                        onToken: (token) => {
                            if (!token) return;
                            setMessages(prev => {
                                const last = prev[prev.length - 1];
                                if (!last || last.role !== 'assistant') return prev;
                                return prev.map((message, index) => index === prev.length - 1
                                    ? { ...message, content: `${message.content || ''}${token}`, statusStep: null }
                                    : message);
                            });
                        },
                        // onStatus — push into statusHistory for step card display
                        onStatus: (event) => {
                            setMessages(prev => {
                                const last = prev[prev.length - 1];
                                if (!last || last.role !== 'assistant') return prev;
                                const label = STATUS_LABELS[event.step] || event.detail || event.step;
                                const newEntry = { step: event.step, label };
                                return prev.map((m, i) => i === prev.length - 1
                                    ? { ...m,
                                        statusStep: event.step,
                                        statusDetail: event.detail || null,
                                        statusHistory: [...(m.statusHistory || []), newEntry] }
                                    : m);
                            });
                        },
                        // onSources
                        onSources: (event) => {
                            const webNote = (typeof event.web_auto_note === 'string' && event.web_auto_note.trim()) || null;
                            const fallbackNote = (typeof event.fallback_note === 'string' && event.fallback_note.trim()) || null;
                            setMessages(prev => {
                                const last = prev[prev.length - 1];
                                if (!last || last.role !== 'assistant') return prev;
                                return prev.map((m, i) => i === prev.length - 1
                                    ? { ...m, sources: event.sources || [], response_type: event.response_type, unverified: event.unverified === true, webAutoNote: webNote, fallbackNote }
                                    : m);
                            });
                        },
                        // onFollowups
                        onFollowups: (questions) => {
                            const normalizedQuestions = normalizeFollowupSuggestions(questions);
                            setMessages(prev => {
                                const last = prev[prev.length - 1];
                                if (!last || last.role !== 'assistant') return prev;
                                return prev.map((m, i) => i === prev.length - 1
                                    ? { ...m, followups: normalizedQuestions }
                                    : m);
                            });
                        },
                        // onUsage
                        onUsage: (event) => {
                            setMessages(prev => {
                                const last = prev[prev.length - 1];
                                if (!last || last.role !== 'assistant') return prev;
                                return prev.map((m, i) => i === prev.length - 1
                                    ? { ...m, usage: { prompt_tokens: event.prompt_tokens, completion_tokens: event.completion_tokens } }
                                    : m);
                            });
                        },
                        // onNotice — one-line server notices (e.g. stale session scope).
                        onNotice: (event) => {
                            if (event && event.message) showToast(event.message, 'info');
                            // Whole-vault fallback: the stored scope is dead, so drop
                            // the chip instead of claiming scope that isn't applied.
                            if (event && event.scope_fallback) {
                                const deadKey = activeChatId || pgChatIdRef.current;
                                if (deadKey) {
                                    setChatScopes(prev => {
                                        if (!prev || !(deadKey in prev)) return prev;
                                        const next = { ...prev };
                                        delete next[deadKey];
                                        return next;
                                    });
                                }
                            }
                        },
                        // onClarification — one-round web-clarification turn.
                        // The server persisted the pending flag; the next
                        // user message answers this question. Render it as
                        // the assistant message content (terminal event, like
                        // error — no further content/sources arrive).
                        onClarification: (event) => {
                            const question = event && typeof event.question === 'string' ? event.question : '';
                            if (!question) return;
                            setMessages(prev => prev.map((m, i) => i === prev.length - 1 && m.role === 'assistant'
                                ? { ...m, streaming: false, content: question, clarification: true, sources: [], followups: [],
                                    usage: null, statusStep: null, statusDetail: null }
                                : m));
                        },
                        chatUploadIds,
                        chatId: activeChatId || null,
                        // onError — set error on assistant message, stop streaming.
                        // Clear any partial answer content so the error card
                        // replaces it (backend error events are terminal in the
                        // read loop — no further content/sources events arrive).
                        onError: (event) => {
                            setMessages(prev => prev.map((m, i) => i === prev.length - 1 && m.role === 'assistant'
                                ? { ...m, streaming: false, content: '', sources: [], followups: [],
                                    usage: null, statusStep: null, statusDetail: null,
                                    error: { code: event.code, category: event.category, message: event.message, label: event.label, hint: event.hint, icon: event.icon, model: event.model, provider: event.provider } }
                                : m));
                        },
                    });
                    requestCompleted = !res?.error;

                    setMessages(prev => prev.map((message, index) => index === prev.length - 1 && message.role === 'assistant'
                        ? { ...message, streaming: false }
                        : message));

                    // Handle confirmation_needed (returned early from SSE)
                    if (res && res.needs_confirmation) {
                        // Remove the empty streaming placeholder
                        setMessages(prev => prev.filter((m, i) => !(i === prev.length - 1 && m.role === 'assistant' && m.streaming === false && !m.content)));

                        const filteredFiles = (res.matched_files || []).filter(f => f.match_percentage > 0);
                        const allIds = filteredFiles.map(f => f.file_id);
                        setMessages(prev => {
                            const msgIndex = prev.length;
                            setSelectedIds(s => ({ ...s, [msgIndex]: allIds }));
                            return [...prev, {
                                role: 'system',
                                content: filteredFiles.length > 0
                                        ? `We have ${res.total_files ?? '?'} files in vault. Found ${filteredFiles.length} relevant. Choose the files to process.`
                                        : `I couldn't find any highly relevant files. Should I proceed without vault context?`,
                                matchedFiles: filteredFiles,
                                msgIndex: msgIndex,
                                originalInput: text
                            }];
                        });
                    }

                    // Refresh session list to pick up LLM-generated title.
                    // Always refresh; delay 3s on first message to let title generation complete.
                    const isFirstMsg = messages.filter(m => m.role === 'user').length <= 1;
                    setTimeout(() => onSessionUpdate?.(), isFirstMsg ? 6000 : 500);

                    // Refresh relationship memory cache for About Me summary
                    api.getGraphMemory().then(data => {
                        if (data) setGraphMemory(data);
                    }).catch(() => {});
                } catch (e) {
                    // Remove empty streaming message on error
                    setMessages(prev => {
                        const last = prev[prev.length - 1];
                        if (last && last.role === 'assistant' && last.streaming && !last.content) {
                            return prev.slice(0, -1);
                        }
                        if (e.name === 'AbortError' && last?.role === 'assistant' && last.streaming) {
                            return prev.map((message, index) => index === prev.length - 1
                                ? { ...message, streaming: false }
                                : message);
                        }
                        return prev;
                    });
                    if (e.name !== 'AbortError') {
                        console.error('[chat send failed]', e);
                        setMessages(prev => [...prev, { role: 'system', content: `Connection failed: ${e.message || 'unknown error'}` }]);
                    }
                } finally {
                    abortRef.current = null;
                    const currentSelection = activeChatSelectionRef.current;
                    const selectionUnchanged = (
                        currentSelection.sessionId === selectionAtSend.sessionId
                        && currentSelection.pgChatId === selectionAtSend.pgChatId
                    );
                    if (selectionUnchanged) {
                        skipHistoryLoadRef.current = createdPgChat
                            ? { sessionId: selectionAtSend.sessionId, pgChatId: createdPgChat.id }
                            : selectionAtSend;
                    }
                    setProcessing(false);
                    setTaggedFiles([]);
                    // Selecting the new durable chat before its first request
                    // completes starts an empty-history load that can overwrite
                    // the live placeholder. Publish it only after the request
                    // (success or failure) and never override a newer selection.
                    if (createdPgChat && selectionUnchanged) {
                        onPgChatCreated?.(createdPgChat);
                    } else if (createdPgChat) {
                        onSessionUpdate?.();
                    }
                }
            };

            const clearHistory = async () => {
                onConfirmAction({
                    title: 'Clear History?',
                    message: 'All messages in this session will be permanently deleted.',
                    onConfirm: async () => {
                        await api.clearChatHistory(sessionId);
                        setMessages([{ role: 'assistant', content: 'History cleared. How can I help?', created_at: new Date().toISOString() }]);
                    }
                });
            };

            const aiProgress = getIndexProgress(aiStats);
            const aiBox = aiBoxState(aiStats);

            return (
                <React.Fragment>
                <div className="flex-1 flex flex-col h-full bg-[#000000] relative">
                    <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/10 to-transparent pointer-events-none"></div>


                    <div className="h-14 border-b border-white/5 flex items-center justify-between px-6 bg-black/40 backdrop-blur-md sticky top-0 z-10">
                        <div className="flex items-center gap-3">
                            <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)] animate-pulse"></div>
                            <h1 className="text-base font-bold gradient-text">Lavix AI Assistant</h1>
                            {aiStats && (
                                <div className="flex flex-col gap-1 ml-2">
                                    <div
                                        onClick={handleAiReadyClick}
                                        className={`flex items-center gap-3 px-3 py-1.5 rounded-lg animate-fade-in transition-all cursor-pointer select-none ${
                                            isIndexing
                                                ? 'bg-amber-500/10 border border-amber-500/30'
                                                : fileSearch
                                                ? 'bg-indigo-600/10 border border-indigo-500/20 hover:bg-indigo-600/20'
                                                : 'bg-white/[0.03] border border-white/10 hover:border-white/20'
                                        }`}
                                        title={isIndexing ? 'Indexing... Click to cancel' : fileSearch ? 'File indexing active — click to manage' : 'Click to enable file indexing'}
                                    >
                                        <div className="flex items-center gap-2">
                                            {isIndexing ? (
                                                <div className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping shadow-[0_0_8px_rgba(251,191,36,0.6)]"></div>
                                            ) : aiBox.failed ? (
                                                <div className="w-1.5 h-1.5 rounded-full bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]"></div>
                                            ) : (
                                                <div className={`w-1.5 h-1.5 rounded-full animate-pulse shadow-[0_0_8px_rgba(129,140,248,0.6)] ${fileSearch ? 'bg-indigo-400' : 'bg-zinc-600'}`}></div>
                                            )}
                                            <span className={`text-[11px] font-black uppercase tracking-widest leading-none ${isIndexing ? 'text-amber-300' : aiBox.failed ? 'text-red-300' : fileSearch ? 'text-indigo-300' : 'text-zinc-500'}`}>
                                                {isIndexing ? 'INDEXING' : aiBox.failed ? 'INDEX FAILED' : 'AI-READY'}
                                            </span>
                                        </div>

                                        <div className="flex flex-col w-28 gap-1">
                                            <div className="flex items-end justify-between">
                                                <span className={`text-[9px] font-bold leading-none ${isIndexing ? 'text-amber-400/80' : 'text-indigo-400/80'}`}>{aiProgress.searchable}/{aiProgress.allFilesTotal}</span>
                                                <span className={`text-[9px] font-bold leading-none ${isIndexing ? 'text-amber-300' : aiBox.failed ? 'text-red-300' : 'text-indigo-300'}`}>{aiProgress.percentage}%</span>
                                            </div>
                                            <div className="w-full h-1 bg-black/40 rounded-full overflow-hidden border border-white/5">
                                                <div
                                                    className={`h-full rounded-full transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(124,58,237,0.3)] ${
                                                        isIndexing
                                                            ? 'bg-gradient-to-r from-amber-500 to-orange-500'
                                                            : 'bg-gradient-to-r from-blue-600 to-violet-500'
                                                    }`}
                                                    style={{ width: `${aiProgress.percentage}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                        {isIndexing && (
                                            <button
                                                onClick={(e) => { e.stopPropagation(); handleAiReadyClick(); }}
                                                className="ml-1 p-1 rounded-full bg-white/10 hover:bg-red-500/20 text-zinc-400 hover:text-red-400 transition-all"
                                                title="Cancel indexing"
                                            >
                                                <Icons.X size={10} />
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="flex items-center gap-2 relative">
                            <div ref={modelMenuRef} className="relative">
                                <button
                                    data-testid="chat-model-menu"
                                    onClick={() => setShowModelMenu(v => !v)}
                                    className={`flex items-center gap-2 px-3 py-2 rounded-xl border transition-all ${showModelMenu ? 'bg-white/10 border-white/20 text-white' : 'bg-white/5 border-white/10 text-zinc-400 hover:bg-white/10 hover:text-white'}`}
                                    title="Model routing"
                                >
                                    <span className="text-[10px] font-black uppercase tracking-[0.18em] text-indigo-300">Model</span>
                                    <span className="text-xs font-medium max-w-[140px] truncate">{modelConfig.model || 'Loading...'}</span>
                                    <svg width="10" height="10" viewBox="0 0 20 20" fill="none" className={`transition-transform ${showModelMenu ? 'rotate-180' : ''}`}>
                                        <path d="M5 7.5L10 12.5L15 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                                    </svg>
                                </button>

                                                {showModelMenu && (
                                    <div className="absolute right-0 top-full mt-2 z-40 w-[400px] max-w-[calc(100vw-2rem)] bg-[#0d0d0f] border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
                                        {/* Header */}
                                        <div className="px-4 py-3 border-b border-white/5">
                                            <div className="flex items-center justify-between gap-3">
                                                <div>
                                                    <div className="text-[10px] font-black uppercase tracking-[0.18em] text-zinc-500">Model Config</div>
                                                    <div className="text-xs text-zinc-400 mt-1">Change model below — applies immediately.</div>
                                                </div>
                                                <button
                                                    onClick={refreshAvailableModels}
                                                    disabled={refreshingModels}
                                                    className={`px-2.5 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-[0.18em] border transition-all ${refreshingModels ? 'bg-indigo-600/20 text-indigo-200 border-indigo-500/30' : 'bg-white/5 text-zinc-400 border-white/10 hover:text-white hover:bg-white/10'}`}
                                                >
                                                    {refreshingModels ? 'Refreshing...' : 'Refresh'}
                                                </button>
                                            </div>
                                        </div>

                                        <div className="p-4 space-y-4">
                                            <div>
                                                <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-[0.18em] mb-1.5">Provider</div>
                                                <div className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white">Ollama (Local)</div>
                                            </div>

                                            {/* Available models: click an entry to make it active immediately. */}
                                            <div>
                                                <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-[0.18em] mb-1.5">Active Model</div>
                                                <div className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white truncate" title={modelConfig.model || ''}>{modelConfig.model || 'Loading...'}</div>
                                            </div>
                                            <div>
                                                <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-[0.18em] mb-1.5">
                                                    Allowed ({modelConfig.available_models.length})
                                                </div>
                                                <div className="max-h-[120px] overflow-y-auto space-y-0.5">
                                                    <button
                                                        onClick={() => updateActiveChatModel('')}
                                                        disabled={modelConfigLoading}
                                                        title="Clear override — follow the system default"
                                                        className="w-full text-left px-3 py-1.5 rounded-lg text-xs font-mono text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-all disabled:opacity-50"
                                                    >
                                                        — System default —
                                                    </button>
                                                    {modelConfig.available_models.length > 0
                                                        ? modelConfig.available_models.map(m => (
                                                            <button
                                                                key={m}
                                                                onClick={() => updateActiveChatModel(m)}
                                                                disabled={modelConfigLoading}
                                                                title={`Use ${m} for this chat`}
                                                                className={`w-full text-left px-3 py-1.5 rounded-lg text-xs font-mono transition-all disabled:opacity-50 ${m === modelConfig.model ? 'bg-indigo-500/30 text-indigo-100 border border-indigo-400/50 font-medium' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5 border border-transparent'}`}
                                                            >
                                                                {m === modelConfig.model ? '✓ ' : ''}{m}
                                                            </button>
                                                        ))
                                                        : <div className="px-3 py-1.5 text-xs text-zinc-500">Backend default</div>}
                                                </div>
                                            </div>

                                            {/* Error */}
                                            {modelConfigError && (
                                                <div className="text-[10px] px-3 py-2 rounded-xl border bg-red-500/10 text-red-400 border-red-500/20">
                                                    {modelConfigError}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                            <button onClick={clearHistory} className="p-2 hover:bg-white/5 rounded-lg text-zinc-500 hover:text-red-400 transition-all" title="Clear History">
                                <Icons.Trash size={18} />
                            </button>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-6 space-y-6 z-0" ref={scrollRef}>
                        {loadingHistory ? (
                            <div className="flex flex-col items-center justify-center h-full gap-4 text-zinc-600 animate-pulse">
                                <Icons.Loader size={32} />
                                <span className="text-[10px] font-black uppercase tracking-[0.2em]">Restoring Neural Context...</span>
                            </div>
                        ) : messages.map((m, i) => (
                            <React.Fragment key={i}>
                            <div
                                data-testid={`chat-message-${m.role}`}
                                data-streaming={m.streaming ? 'true' : 'false'}
                                data-message-content={m.content || ''}
                                data-status-history={(m.statusHistory || []).map(item => item.step).join(',')}
                                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`group relative max-w-3xl p-4 rounded-2xl backdrop-blur-md shadow-lg ${m.role === 'user'
                                    ? 'bg-indigo-600/20 border border-indigo-500/20 text-white rounded-tr-sm'
                                    : m.role === 'system' && m.type === 'upload-success'
                                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                        : m.role === 'system' && m.type === 'upload-warn'
                                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                        : m.role === 'system' && m.matchedFiles !== undefined
                                        ? 'glass-card border-white/10 text-gray-300 rounded-tl-sm'
                                        : m.role === 'system'
                                        ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                        : 'glass-card border-white/10 text-gray-300 rounded-tl-sm'
                                    }`}>
                                    {/* Tagged Files in user messages */}
                                    {m.role === 'user' && m.taggedFiles && m.taggedFiles.length > 0 && (
                                        <div className="flex flex-wrap gap-1.5 mb-2 pb-2 border-b border-white/10">
                                            {m.taggedFiles.map((fname, fi) => (
                                                <span key={fi} className="inline-flex items-center gap-1.5 bg-indigo-500/20 border border-indigo-500/30 text-indigo-200 text-xs font-medium px-2.5 py-1 rounded-full">
                                                    <Icons.File size={11} className="text-indigo-400" />
                                                    {fname}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                    {/* OpenWebUI-style status history — shows step progression */}
                                    {m.streaming && !m.content && (
                                        <div className="flex flex-col gap-0.5 py-1">
                                            {/* Completed steps (all except last) */}
                                            {m.statusHistory && m.statusHistory.slice(0, -1).map((step, si) => (
                                                <div key={si} className="status-done-step">
                                                    <div className="step-check"></div>
                                                    <span>{step.label}</span>
                                                </div>
                                            ))}
                                            {/* Active step with shimmer + cycling text */}
                                            {(() => {
                                                const hist = m.statusHistory;
                                                const activeStep = hist && hist.length > 0 ? hist[hist.length - 1] : null;
                                                const msgs = activeStep
                                                    ? (STEP_MSGS_MAP[activeStep.step] || [activeStep.label])
                                                    : THINKING_MSGS;
                                                const label = msgs[statusTick % msgs.length];
                                                return (
                                                    <div className="status-active-step">
                                                        <div className="step-pulse-dot"></div>
                                                        <span className="status-shimmer">{label}</span>
                                                    </div>
                                                );
                                            })()}
                                        </div>
                                    )}
                                    {/* Message content — markdown during streaming AND after (no layout shift) */}
                                    {m.content && (
                                        m.streaming ? (
                                            <div
                                                data-testid="assistant-message-content"
                                                className="text-sm leading-relaxed prose"
                                                style={{ fontSize: 'var(--chat-font-size)' }}
                                                dangerouslySetInnerHTML={{ __html: renderAssistantMarkdown(m.content, m.sources, `message-${i}-source`) + '<span class="streaming-cursor"></span>' }}
                                            />
                                        ) : (
                                            <div
                                                data-testid={m.role === 'assistant' ? 'assistant-message-content' : 'user-message-content'}
                                                className="text-sm leading-relaxed prose"
                                                style={{ fontSize: 'var(--chat-font-size)' }}
                                                dangerouslySetInnerHTML={{ __html: m.role === 'assistant' ? renderAssistantMarkdown(m.content, m.sources, `message-${i}-source`) : renderMarkdown(m.content) }}
                                            />
                                        )
                                    )}
                                    {/* Unverified banner — explanatory answers with no retrievable evidence */}
                                    {m.role === 'assistant' && m.unverified === true && (
                                        <div data-testid="assistant-unverified-banner" className="mt-2.5 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/25 text-amber-300/90 text-[11px]">
                                            <Icons.AlertTriangle size={12} className="flex-shrink-0" />
                                            <span>Background knowledge — not verified against your files or the web.</span>
                                        </div>
                                    )}
                                    {/* Web auto-check note — override fired despite toggle */}
                                    {m.role === 'assistant' && m.webAutoNote && (
                                        <div data-testid="assistant-webauto-note" className="mt-2.5 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/25 text-sky-300/90 text-[11px]">
                                            <Icons.Globe size={12} className="flex-shrink-0" />
                                            <span>{m.webAutoNote}</span>
                                        </div>
                                    )}
                                    {/* Fallback model note — admin fallback served this answer */}
                                    {m.role === 'assistant' && m.fallbackNote && (
                                        <div data-testid="assistant-fallback-note" className="mt-2.5 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-violet-500/10 border border-violet-500/25 text-violet-300/90 text-[11px]">
                                            <Icons.Info size={12} className="flex-shrink-0" />
                                            <span>{m.fallbackNote}</span>
                                        </div>
                                    )}
                                    {/* Sources — show as soon as they arrive (even while still streaming) */}
                                    {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                                        <div data-testid="assistant-sources" className="mt-3 pt-2.5 border-t border-white/[0.06]">
                                            <div className="flex flex-wrap items-center gap-1.5">
                                                <span className="text-[7px] font-bold text-zinc-600 uppercase tracking-[0.15em] mr-0.5">Sources</span>
                                                {m.sources.map((src, j) => {
                                                    const isWeb = src.kind === 'web' || src.mime_type === 'web';
                                                    const sourceUrl = isWeb ? safeExternalUrl(src.url) : null;
                                                    const domain = sourceUrl
                                                        ? (() => { try { return new URL(sourceUrl).hostname.replace('www.', ''); } catch(e) { return 'web'; } })()
                                                        : null;
                                                    const label = domain || src.filename?.split('.').slice(0, -1).join('.') || src.filename || `file`;
                                                    const matchPercentage = resolveMatchPercentage(src);
                                                    const citationId = /^[VW]\d+$/i.test(String(src.id || '')) ? String(src.id).toLowerCase() : `item-${j + 1}`;
                                                    const anchorId = `message-${i}-source-${citationId}`;
                                                    return sourceUrl ? (
                                                        <a
                                                            key={citationId}
                                                            id={anchorId}
                                                            href={sourceUrl}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            title={src.filename}
                                                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] hover:border-emerald-500/30 hover:bg-emerald-500/5 hover:text-emerald-300 text-zinc-400 transition-all text-[10px] max-w-[220px] group"
                                                        >
                                                            <Icons.Globe size={9} className="text-emerald-500/70 flex-shrink-0" />
                                                            <span className="truncate">{label}</span>
                                                            {matchPercentage != null && m.sources.length > 1 && <span className="text-[9px] text-emerald-500/70 flex-shrink-0">{matchPercentage}%</span>}
                                                            <Icons.ArrowUpRight size={8} className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                                                        </a>
                                                    ) : !isWeb ? (
                                                        <button
                                                            key={citationId}
                                                            id={anchorId}
                                                            onClick={() => onPreviewFile(src)}
                                                            title={src.filename}
                                                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] hover:border-indigo-500/30 hover:bg-indigo-500/5 hover:text-indigo-300 text-zinc-400 transition-all text-[10px] max-w-[220px] group"
                                                        >
                                                            <Icons.File size={9} className="text-indigo-400/70 flex-shrink-0" />
                                                            <span className="truncate">{label}</span>
                                                            {matchPercentage != null && m.sources.length > 1 && (
                                                                <span className="text-[9px] text-indigo-500/70 flex-shrink-0">{matchPercentage}%</span>
                                                            )}
                                                        </button>
                                                    ) : (
                                                        <span key={citationId} id={anchorId} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-zinc-500 text-[10px] max-w-[220px]">
                                                            <Icons.Globe size={9} className="flex-shrink-0" />
                                                            <span className="truncate">{label}</span>
                                                            {matchPercentage != null && m.sources.length > 1 && <span className="text-[9px] flex-shrink-0">{matchPercentage}%</span>}
                                                        </span>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                    {/* Token usage + timestamp on same row */}
                                    {((m.role === 'assistant' && !m.streaming && m.usage?.prompt_tokens != null) || (prefTimestamps === true && !m.streaming && m?.created_at)) && (
                                        <div className="mt-2 flex items-center text-[10px] font-mono">
                                            {m.role === 'assistant' && !m.streaming && m.usage?.prompt_tokens != null && (
                                                <span className="text-zinc-600">↑ {m.usage.prompt_tokens} in · ↓ {m.usage.completion_tokens ?? 0} out</span>
                                            )}
                                            {prefTimestamps === true && !m.streaming && m?.created_at && (
                                                <span className="ml-auto text-zinc-700">{new Date(m.created_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                                            )}
                                        </div>
                                    )}
                                    {/* Error banner — shown when LLM returns an error instead of answer */}
                                    {m.role === 'assistant' && m.error && !m.streaming && (
                                        <div className={`mt-3 rounded-xl border p-3 ${
                                            m.error.category === 'payment_required'
                                                ? 'bg-amber-500/15 border-amber-500/30 text-amber-300'
                                                : m.error.category === 'unreachable'
                                                ? 'bg-red-500/15 border-red-500/30 text-red-300'
                                                : m.error.category === 'rate_limited'
                                                ? 'bg-orange-500/15 border-orange-500/30 text-orange-300'
                                                : 'bg-red-500/15 border-red-500/30 text-red-300'
                                        }`}>
                                            <div className="flex items-start gap-2.5">
                                                <span className="text-lg flex-shrink-0">{m.error.icon || '⚠️'}</span>
                                                <div className="flex-1 min-w-0">
                                                    <div className="text-xs font-bold uppercase tracking-wider mb-0.5">
                                                        {m.error.label || 'Error'}
                                                    </div>
                                                    <div className="text-xs opacity-90">{m.error.hint || m.error.message}</div>
                                                    {m.error.model && (
                                                        <div className="text-[10px] opacity-60 mt-1 font-mono">
                                                            {m.error.provider && `${m.error.provider} / `}{m.error.model}
                                                            {m.error.code > 0 && ` (HTTP ${m.error.code})`}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                    {/* Hover action buttons */}
                                    {m.role === 'assistant' && !m.streaming && m.content && (
                                        <div className="absolute -bottom-7 right-0 opacity-0 group-hover:opacity-100 flex gap-1 transition-opacity z-10">
                                            <button
                                                onClick={(e) => copyMessage(assistantDisplayText(m.content, m.sources), e.currentTarget)}
                                                className="flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-900 border border-white/10 text-zinc-500 hover:text-white text-[10px] transition-all"
                                                title="Copy"
                                            >
                                                <Icons.Copy size={10}/> Copy
                                            </button>
                                            <button
                                                onClick={() => regenerateMessage(i)}
                                                className="flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-900 border border-white/10 text-zinc-500 hover:text-indigo-400 text-[10px] transition-all"
                                                title="Regenerate"
                                            >
                                                <Icons.RotateCcw size={10}/> Regen
                                            </button>
                                        </div>
                                    )}
                                    {m.matchedFiles && (
                                        <div className="mt-4 space-y-2 max-w-sm">
                                            {m.matchedFiles.length > 0 && <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-2 px-1">Suggested Files</div>}
                                            {m.matchedFiles.map(file => (
                                                <div key={file.file_id} className="flex items-center gap-2 bg-black/20 p-2 rounded-xl border border-white/5 group hover:bg-black/40 transition-all">
                                                    <input
                                                        type="checkbox"
                                                        className="w-4 h-4 rounded border-white/10 bg-white/5 text-indigo-500 focus:ring-indigo-500/20"
                                                        checked={selectedIds[m.msgIndex]?.includes(file.file_id)}
                                                        onChange={() => toggleFile(m.msgIndex, file.file_id)}
                                                    />
                                                    <button
                                                        onClick={() => onPreviewFile(file)}
                                                        className="flex-1 text-left text-xs text-indigo-300 hover:text-indigo-100 line-clamp-2 break-all transition-colors"
                                                    >
                                                        {file.filename}
                                                    </button>
                                                    <div className="text-[10px] text-zinc-600 font-mono">{file.match_percentage}%</div>
                                                </div>
                                            ))}
                                            {!resolvedMsgs[m.msgIndex] && (
                                                <div className="flex gap-2 mt-4">
                                                    <button
                                                        data-testid="chat-analyze-selected"
                                                        onClick={() => {
                                                            const ids = selectedIds[m.msgIndex] || [];
                                                            if (ids.length === 0 && m.matchedFiles.length > 0) return alert('Select at least one file');
                                                            const merged = ids.length > 0
                                                                ? unionScopeIds(ids, normalizeScopeIds(chatScopes[pgChatId || pgChatIdRef.current] || []))
                                                                : null;
                                                            pendingScopeRef.current = merged === null ? null : [...merged];
                                                            pendingFolderScopeRef.current = null;
                                                            send(m.originalInput, true, merged || [], m.msgIndex, sessionFolderIds());
                                                        }}
                                                        className="flex-1 text-[10px] font-black uppercase tracking-[0.22em] bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white px-4 py-2.5 rounded-xl transition-all flex items-center justify-center gap-2"
                                                    >
                                                        <Icons.Zap size={12} />
                                                        {m.matchedFiles.length > 0 ? 'Process Selected' : 'Proceed'}
                                                    </button>
                                                    <button
                                                        onClick={() => send(m.originalInput, false, followupFileIds(chatScopes[pgChatId || pgChatIdRef.current] || []), m.msgIndex, sessionFolderIds())}
                                                        className="px-4 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-[0.22em] bg-white/5 text-zinc-400 border border-white/10 hover:bg-white/10 hover:text-zinc-200 transition-all"
                                                    >
                                                        Skip
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                    {m.action && (
                                        <button
                                            onClick={m.action}
                                            className="mt-3 text-xs font-semibold bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                                        >
                                            <Icons.Zap size={14} />
                                            {m.actionLabel}
                                        </button>
                                    )}
                                </div>
                            </div>
                            {/* Follow-up suggestions — outside the bubble */}
                            {m.role === 'assistant' && !m.streaming && m.followups && m.followups.length > 0 && (
                                <div data-testid="assistant-followups" className="flex flex-wrap gap-1.5 mt-2 ml-1 max-w-3xl">
                                    {m.followups.map((q, fi) => (
                                        <button
                                            key={fi}
                                            onClick={() => !processing && sendFollowup(q)}
                                            className="text-[11px] text-left leading-snug text-zinc-500 border border-white/[0.07] rounded-xl px-3 py-1.5 hover:border-indigo-500/40 hover:text-zinc-300 hover:bg-indigo-500/5 transition-all disabled:opacity-40 max-w-[min(24rem,calc(100vw-3rem))] line-clamp-2"
                                            disabled={processing}
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            )}
                            </React.Fragment>
                        ))}
                        {/* Grant Access — modal overlay */}
                        {grantPromptFiles && !grantDone && (
                            <div className="flex justify-start animate-fade-in">
                                <div className="max-w-[400px] w-full bg-[#0c0c0e] border border-amber-500/20 rounded-2xl rounded-tl-sm overflow-hidden shadow-lg">
                                    <div className="p-4">
                                        <div className="flex items-center gap-2.5 mb-3">
                                            <div className="w-7 h-7 rounded-full bg-amber-500/15 flex items-center justify-center text-amber-400 flex-shrink-0">
                                                <Icons.Lock size={13} />
                                            </div>
                                            <div>
                                                <p className="text-[13px] font-bold text-white leading-none">Indexing Access Required</p>
                                                <p className="text-[11px] text-zinc-500">Lavix needs permission to index {grantPromptFiles.length === 1 ? 'this file' : 'these files'} before answering</p>
                                            </div>
                                        </div>
                                        <div className="space-y-1.5 mb-3 max-h-[120px] overflow-y-auto">
                                            {grantPromptFiles.map(f => (
                                                <div key={f.id} className="flex items-center gap-2 bg-amber-500/5 border border-amber-500/15 rounded-xl px-3 py-2">
                                                    <Icons.File size={12} className="text-amber-400 flex-shrink-0" />
                                                    <span className="text-xs text-zinc-300 truncate flex-1">{f.filename}</span>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={() => { setGrantPromptFiles(null); setPendingSend(null); }}
                                                className="flex-1 py-2 text-xs font-semibold text-zinc-400 hover:text-white border border-white/10 rounded-xl transition-colors"
                                            >
                                                Cancel
                                            </button>
                                            <button
                                                onClick={handleGrantAndSend}
                                                className="flex-1 py-2 text-xs font-semibold text-indigo-300 hover:text-indigo-200 bg-indigo-500/10 border border-indigo-500/20 rounded-xl transition-colors flex items-center justify-center gap-2"
                                            >
                                                <Icons.Shield size={12} /> Grant &amp; Ask
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        {grantDone && (
                            <div className="flex justify-start animate-fade-in">
                                <div className="max-w-[400px] w-full bg-[#0c0c0e] border border-emerald-500/20 rounded-2xl rounded-tl-sm overflow-hidden shadow-lg">
                                    <div className="p-4">
                                        <div className="flex items-center gap-2.5 mb-2">
                                            <div className="w-7 h-7 rounded-full bg-emerald-500/15 flex items-center justify-center text-emerald-400 flex-shrink-0">
                                                <Icons.Check size={13} />
                                            </div>
                                            <div>
                                                <p className="text-[13px] font-bold text-white leading-none">Access granted</p>
                                                <p className="text-[11px] text-zinc-500">Vector indexing is in progress…</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        {processing && !messages.some(m => m.streaming) && (
                            <div className="flex justify-start">
                                <div className="group relative glass-card border-white/10 px-5 py-3 rounded-2xl rounded-tl-sm flex items-center gap-2.5 cursor-default">
                                    <div className="step-pulse-dot"></div>
                                    <span className="status-shimmer text-xs group-hover:opacity-0 transition-opacity duration-150">
                                        {THINKING_MSGS[statusTick % THINKING_MSGS.length]}
                                    </span>
                                    <button
                                        onClick={cancelRequest}
                                        className="absolute inset-0 flex items-center justify-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-150 text-[11px] font-bold text-red-400 hover:text-red-300 uppercase tracking-wide rounded-2xl rounded-tl-sm"
                                    >
                                        <Icons.X size={11} /> Cancel
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="p-8 pb-10 z-10">
                        <div className="max-w-4xl mx-auto">
                            {/* Tagged Files Pills */}
                            {(() => {
                                const scopeKey = pgChatId || pgChatIdRef.current;
                                const sessionScope = normalizeScopeIds(scopeKey ? chatScopes[scopeKey] : []);
                                const sessionFolders = normalizeScopeIds(scopeKey ? chatFolderScopes[scopeKey] : []);
                                const hasScope = sessionScope.length > 0 || sessionFolders.length > 0;
                                if (taggedFiles.length === 0 && taggedFolders.length === 0 && !hasScope) return null;
                                // Pill shows the COMMITTED session scope only — never a
                                // preview of uncommitted composer tags. The manager
                                // reads the same state, so pill and popup agree.
                                const scopeNames = sessionScope.map(id => {
                                    const f = files.find(ff => ff.id === id);
                                    return f ? (f.original_filename || f.filename) : `File #${id}`;
                                });
                                return (
                                    <>
                                    <div className="flex flex-wrap items-center gap-2 mb-3 animate-fade-in">
                                        {taggedFiles.filter(tf => isImageFilename(tf.filename) && !thumbUrls[tf.id] && !thumbFailed[tf.id]).map(tf => (
                                            <div key={`shimmer-${tf.id}`} className="relative flex-shrink-0">
                                                <div title={`${tf.filename} · indexing…`} className="w-16 h-16 rounded-xl border border-amber-500/30 bg-amber-500/5 overflow-hidden">
                                                    <div className="w-full h-full animate-pulse bg-gradient-to-br from-amber-500/10 via-white/5 to-amber-500/10" />
                                                </div>
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); removeTaggedFile(tf.id); }}
                                                    title={`Untag ${tf.filename}`}
                                                    className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-zinc-900/90 hover:bg-red-500/60 border border-white/10 flex items-center justify-center text-zinc-400 hover:text-white transition-all"
                                                >
                                                    <Icons.X size={10} />
                                                </button>
                                            </div>
                                        ))}
                                        {taggedFiles.filter(tf => thumbUrls[tf.id]).map(tf => (
                                            <div key={`thumb-${tf.id}`} className="relative group flex-shrink-0">
                                                <img
                                                    src={thumbUrls[tf.id]}
                                                    alt={tf.filename}
                                                    title={tf.filename}
                                                    onClick={() => onPreviewFile({ id: tf.id, filename: tf.filename })}
                                                    className="w-16 h-16 object-cover rounded-xl border border-indigo-500/30 cursor-pointer hover:border-indigo-400/60 transition-all"
                                                />
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); removeTaggedFile(tf.id); }}
                                                    title={`Untag ${tf.filename}`}
                                                    className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-zinc-900/90 hover:bg-red-500/60 border border-white/10 flex items-center justify-center text-zinc-400 hover:text-white transition-all"
                                                >
                                                    <Icons.X size={10} />
                                                </button>
                                            </div>
                                        ))}
                                        {processing && (taggedFiles.length > 0 || taggedFolders.length > 0) && (
                                            <span className="w-full text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-300/80">
                                                Currently reading…
                                            </span>
                                        )}
                                        {/* Folder pill at position 0, scoop pill at position 1.
                                            Folders are never grouped into the scoop pill. */}
                                        {(() => {
                                            const folderIds = [];
                                            for (const id of [...sessionFolders, ...taggedFolders.map(f => f.id)]) {
                                                if (!folderIds.includes(id)) folderIds.push(id);
                                            }
                                            if (folderIds.length === 0) return null;
                                            const folderNames = folderIds.map(id => {
                                                const f = folders.find(ff => ff.id === id);
                                                if (f) return f.name;
                                                const tf = taggedFolders.find(t => t.id === id);
                                                return tf ? tf.filename : `Folder #${id}`;
                                            });
                                            return (
                                                <div
                                                    onClick={() => setShowScopeManager(true)}
                                                    title="Manage folder scope"
                                                    className="flex items-center gap-2 border border-emerald-500/30 bg-emerald-600/10 hover:bg-emerald-600/20 rounded-full pl-3 pr-1.5 py-1 cursor-pointer transition-all order-first animate-fade-in"
                                                >
                                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" className="text-emerald-400 flex-shrink-0"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                    <span
                                                        className="text-xs font-medium max-w-[260px] truncate text-emerald-200"
                                                        title={folderNames.join(', ')}
                                                    >
                                                        Scoped to {folderIds.length} folder{folderIds.length === 1 ? '' : 's'}: {folderNames.slice(0, 2).join(', ')}{folderIds.length > 2 ? '…' : ''}
                                                    </span>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            clearFolderScopeNow();
                                                        }}
                                                        title="Clear folder scope — file scope untouched"
                                                        className="w-5 h-5 rounded-full bg-white/5 hover:bg-red-500/30 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-all"
                                                    >
                                                        <Icons.X size={10} />
                                                    </button>
                                                </div>
                                            );
                                        })()}
                                        {sessionScope.length > 0 && (
                                            <div
                                                onClick={() => setShowScopeManager(true)}
                                                title="Manage chat file scope"
                                                className="flex items-center gap-2 border border-indigo-500/30 bg-indigo-600/15 hover:bg-indigo-600/25 rounded-full pl-3 pr-1.5 py-1 cursor-pointer transition-all transition-all"
                                            >
                                                <Icons.File size={12} className="text-indigo-400 flex-shrink-0" />
                                                <span
                                                    className="text-xs font-medium max-w-[260px] truncate text-indigo-200"
                                                    title={scopeNames.join(', ')}
                                                >
                                                    Scoped to {sessionScope.length} file{sessionScope.length === 1 ? '' : 's'}: {scopeNames.slice(0, 2).join(', ')}{sessionScope.length > 2 ? '…' : ''}
                                                </span>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        clearScopeNow();
                                                    }}
                                                    title="Clear file scope — next message searches everything"
                                                    className="w-5 h-5 rounded-full bg-white/5 hover:bg-red-500/30 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-all"
                                                >
                                                    <Icons.X size={10} />
                                                </button>
                                            </div>
                                        )}
                                        {(() => {
                                            // Advisory scope-mismatch notice: the input names
                                            // specific entities (capitalized/digit tokens) with
                                            // zero overlap against scoped files' filenames/tags.
                                            // Advisory only — never blocks, never auto-clears.
                                            // Short or generic inputs stay silent.
                                            const key = pgChatId || pgChatIdRef.current;
                                            const scoped = normalizeScopeIds(key ? chatScopes[key] : []);
                                            if (scoped.length === 0) return null;
                                            const stop = new Set(['what', 'which', 'that', 'this', 'have', 'with', 'from', 'about', 'your', 'there', 'their', 'when', 'where', 'tell', 'does', 'file', 'files', 'show', 'list', 'give', 'much', 'many', 'have', 'has', 'had', 'and', 'the']);
                                            const words = ((typeof input === 'string' ? input : '').match(/[A-Za-z0-9][A-Za-z0-9'\-]{2,}/g) || []);
                                            const entities = [...new Set(words.filter(w => (/[A-Z]/.test(w) || /\d/.test(w)) && !stop.has(w.toLowerCase())))];
                                            if (entities.length === 0) return null;
                                            const scopeWords = new Set();
                                            scoped.forEach(id => {
                                                const f = files.find(ff => ff.id === id);
                                                if (!f) return;
                                                (((f.original_filename || f.filename) || '') + ' ' + ((f.tags || []).join(' '))).toLowerCase().split(/[^a-z0-9]+/).forEach(w => { if (w.length > 2) scopeWords.add(w); });
                                            });
                                            if (entities.some(e => scopeWords.has(e.toLowerCase()))) return null;
                                            const scopedNames = scoped.map(id => {
                                                const f = files.find(ff => ff.id === id);
                                                return f ? (f.original_filename || f.filename) : `File #${id}`;
                                            });
                                            return (
                                                <div
                                                    data-testid="scope-mismatch-notice"
                                                    className="flex items-center gap-2 border border-amber-500/25 bg-amber-500/[0.07] hover:bg-amber-500/[0.12] rounded-full pl-3 pr-1.5 py-1 transition-all animate-fade-in"
                                                >
                                                    <span
                                                        className="text-xs font-medium max-w-[300px] truncate text-amber-200/90"
                                                        title={`Still scoped to: ${scopedNames.join(', ')}`}
                                                    >
                                                        Still scoped to {scoped.length} file{scoped.length === 1 ? '' : 's'} — may not be relevant
                                                    </span>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            clearScopeNow();
                                                        }}
                                                        title="Clear file scope — next message searches everything"
                                                        className="flex-shrink-0 text-[11px] font-bold text-amber-300 hover:text-white bg-amber-500/10 hover:bg-amber-500/25 border border-amber-500/25 rounded-full px-2.5 py-0.5 transition-all"
                                                    >
                                                        Clear scope
                                                    </button>
                                                </div>
                                            );
                                        })()}
                                        {taggedFiles
                                            // Middle-only dedupe: a file already named by the
                                            // committed scoop pill never renders a second chip
                                            // beside it (e.g. discovery re-tagging mid-run).
                                            // Images never render pills: ready ones show
                                            // thumbnails above, indexing ones a shimmer,
                                            // failed ones fall back to a pill.
                                            // State is untouched — display only.
                                            .filter(tf => !sessionScope.includes(tf.id))
                                            .filter(tf => !isImageFilename(tf.filename) || thumbFailed[tf.id])
                                            .map(tf => {
                                        const isIndexing = indexingTagIds.has(tf.id);
                                        const warned = !isIndexing && tf._warn;
                                        return (
                                            <div
                                                key={tf.id}
                                                title={warned ? tf._warn : undefined}
                                                className={`flex items-center gap-2 border rounded-full pl-3 pr-1.5 py-1 group transition-all animate-fade-in ${isIndexing ? 'bg-amber-500/10 border-amber-500/30 hover:bg-amber-500/20' : warned ? 'bg-red-500/10 border-red-500/30 hover:bg-red-500/20' : 'bg-indigo-600/15 border-indigo-500/30 hover:bg-indigo-600/25'}`}
                                            >
                                                {isIndexing ? (
                                                    <svg className="animate-spin text-amber-400 flex-shrink-0" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
                                                ) : (
                                                    <Icons.File size={12} className={`${warned ? 'text-red-400' : 'text-indigo-400'} flex-shrink-0`} />
                                                )}
                                                <span className={`text-xs font-medium max-w-[180px] truncate ${isIndexing ? 'text-amber-200' : warned ? 'text-red-200' : 'text-indigo-200'}`}>
                                                    {tf.filename}{isIndexing ? ' · indexing…' : warned ? ' · not indexed' : ''}
                                                </span>
                                                <button
                                                    onClick={() => removeTaggedFile(tf.id)}
                                                    className="w-5 h-5 rounded-full bg-white/5 hover:bg-red-500/30 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-all"
                                                >
                                                    <Icons.X size={10} />
                                                </button>
                                            </div>
                                        );
                                    })}
                                    </div>
                                    {/* Scope manager popup — the ONLY new onclick lives on
                                        the scope pill above; this panel only closes. */}
                                    {showScopeManager && (hasScope || taggedFolders.length > 0) && (
                                        <div
                                            className="fixed inset-0 z-[70] flex items-end justify-center pb-28 bg-black/40"
                                            onClick={() => setShowScopeManager(false)}
                                        >
                                            <div
                                                className="w-full max-w-md bg-[#121214] border border-white/10 rounded-2xl shadow-2xl overflow-hidden animate-fade-in"
                                                onClick={(e) => e.stopPropagation()}
                                            >
                                                <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                                                    <span className="text-xs font-bold uppercase tracking-[0.18em] text-indigo-300">
                                                        Chat scope — {sessionScope.length + sessionFolders.length} item{(sessionScope.length + sessionFolders.length) === 1 ? '' : 's'}
                                                    </span>
                                                    <button
                                                        onClick={() => setShowScopeManager(false)}
                                                        className="w-6 h-6 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center text-zinc-400 hover:text-zinc-200 transition-all"
                                                    >
                                                        <Icons.X size={12} />
                                                    </button>
                                                </div>
                                                <div className="max-h-64 overflow-y-auto p-2">
                                                    {taggedFolders
                                                        .filter(tf => ![...sessionFolders].includes(tf.id))
                                                        .map(tf => (
                                                            <div key={`pending-folder-${tf.id}`} className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/5 hover:bg-emerald-500/10 transition-all">
                                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className="text-emerald-400 flex-shrink-0"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                                <span className="text-sm text-zinc-200 truncate flex-1" title={`${tf.filename} (tagged — joins scope on send)`}>{tf.filename} <span className="text-[10px] text-emerald-400/80">just tagged</span></span>
                                                                <button
                                                                    onClick={() => removeTaggedFolder(tf.id)}
                                                                    title={`Untag folder ${tf.filename}`}
                                                                    className="w-6 h-6 rounded-full bg-white/5 hover:bg-red-500/30 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-all flex-shrink-0"
                                                                >
                                                                    <Icons.X size={11} />
                                                                </button>
                                                            </div>
                                                        ))}
                                                    {sessionFolders.map(id => {
                                                        const f = folders.find(ff => ff.id === id);
                                                        const name = f ? f.name : `Folder #${id}`;
                                                        return (
                                                            <div key={`folder-${id}`} className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-white/5 transition-all">
                                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className="text-emerald-400 flex-shrink-0"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                                <span className="text-sm text-zinc-200 truncate flex-1" title={`${name} (live folder — new uploads join automatically)`}>{name} <span className="text-[10px] text-emerald-400/80">live folder</span></span>
                                                                <button
                                                                    onClick={() => removeScopeFolder(id)}
                                                                    title={`Remove folder ${name} from chat scope`}
                                                                    className="w-6 h-6 rounded-full bg-white/5 hover:bg-red-500/30 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-all flex-shrink-0"
                                                                >
                                                                    <Icons.X size={11} />
                                                                </button>
                                                            </div>
                                                        );
                                                    })}
                                                    {sessionScope.map(id => {
                                                        const f = files.find(ff => ff.id === id);
                                                        const name = f ? (f.original_filename || f.filename) : `File #${id}`;
                                                        return (
                                                            <div key={id} className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-white/5 transition-all">
                                                                <Icons.File size={14} className="text-indigo-400 flex-shrink-0" />
                                                                <span className="text-sm text-zinc-200 truncate flex-1" title={name}>{name}</span>
                                                                <button
                                                                    onClick={() => removeScopeFile(id)}
                                                                    title={`Remove ${name} from chat scope`}
                                                                    className="w-6 h-6 rounded-full bg-white/5 hover:bg-red-500/30 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-all flex-shrink-0"
                                                                >
                                                                    <Icons.X size={11} />
                                                                </button>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                                <div className="px-4 py-3 border-t border-white/5 flex items-center justify-between">
                                                    <span className="text-[11px] text-zinc-500">Type @ in chatbar to add more files</span>
                                                    <div className="flex gap-2">
                                                        <button
                                                            onClick={() => clearScopeNow()}
                                                            className="px-3 py-1.5 rounded-xl text-[10px] font-bold uppercase tracking-wider text-zinc-400 hover:text-red-300 hover:bg-red-500/10 transition-all"
                                                        >
                                                            Clear all
                                                        </button>
                                                        <button
                                                            onClick={() => setShowScopeManager(false)}
                                                            className="px-4 py-1.5 rounded-xl text-[10px] font-bold uppercase tracking-wider bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white transition-all"
                                                        >
                                                            Done
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                    </>
                                );
                            })()}

                            <div className="relative">
                                {/* @Mention Popup */}
                                {mentionActive && mentionResults.total > 0 && (
                                    <div
                                        ref={mentionPopupRef}
                                        className="absolute bottom-full mb-2 left-0 right-0 bg-[#0c0c0d] border border-white/10 rounded-2xl shadow-2xl shadow-black/50 overflow-hidden z-50 animate-fade-in backdrop-blur-xl"
                                    >
                                        <div className="px-4 py-2.5 border-b border-white/5 flex items-center justify-between">
                                            <div className="flex items-center gap-1">
                                                <button
                                                    onClick={() => { setMentionTab('recent'); setMentionIndex(0); }}
                                                    className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-md transition-all ${mentionTab === 'recent'
                                                        ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-400/30'
                                                        : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
                                                        }`}
                                                >Recent</button>
                                                <button
                                                    onClick={() => { setMentionTab('files'); setMentionIndex(0); }}
                                                    className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-md transition-all ${mentionTab === 'files'
                                                        ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-400/30'
                                                        : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
                                                        }`}
                                                >Files</button>
                                                <button
                                                    onClick={() => { setMentionTab('folders'); setMentionIndex(0); }}
                                                    className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-md transition-all ${mentionTab === 'folders'
                                                        ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-400/30'
                                                        : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
                                                        }`}
                                                >Folders</button>
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <kbd className="text-[8px] bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-zinc-500 font-mono">↑↓</kbd>
                                                <kbd className="text-[8px] bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-zinc-500 font-mono">Enter</kbd>
                                                <kbd className="text-[8px] bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-zinc-500 font-mono">Tab</kbd>
                                                <kbd className="text-[8px] bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-zinc-500 font-mono">Esc</kbd>
                                            </div>
                                        </div>
                                        <div className="max-h-[264px] overflow-y-auto">
                                            {(mentionTab === 'files' || mentionTab === 'recent') && (
                                                <>
                                                    {mentionResults.folderCount > 0 && mentionResults.folders.map((file, idx) => (
                                                        idx === 0 && <div key="folder-header" className="px-4 pt-2 pb-1 text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em]">Folders</div>
                                                    ))}
                                                    {mentionResults.folders.map((file, idx) => {
                                                        const flatIdx = mentionResults.flatIndexMap[file.id] ?? idx;
                                                        const isExpanded = expandedFolderId === file._folderId;
                                                        return (
                                                        <React.Fragment key={file.id}>
                                                        <button
                                                            data-mention-idx={flatIdx}
                                                            onClick={() => selectMention(file)}
                                                            onMouseEnter={() => setMentionIndex(flatIdx)}
                                                            className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-all ${flatIdx === mentionIndex
                                                                ? 'bg-indigo-600/15 border-l-2 border-indigo-500'
                                                                : 'hover:bg-white/5 border-l-2 border-transparent'
                                                                }`}
                                                        >
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); selectMentionFolder(file); }}
                                                                title="Scope whole folder (live)"
                                                                className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded text-emerald-400 bg-emerald-500/15 hover:bg-emerald-500/30 transition-all flex-shrink-0"
                                                            >
                                                                Tag
                                                            </button>
                                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${flatIdx === mentionIndex ? 'bg-indigo-600/20 text-indigo-400' : 'bg-indigo-500/10 text-indigo-500'}`}>
                                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className={`transition-transform ${isExpanded ? 'rotate-90' : ''}`}><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                            </div>
                                                            <div className="flex-1 min-w-0">
                                                                <p className={`text-sm font-medium truncate ${flatIdx === mentionIndex ? 'text-white' : 'text-zinc-300'}`}>
                                                                    {file.filename}
                                                                </p>
                                                                <p className="text-[10px] text-zinc-600 truncate" title={folderAbsolutePath(file._folderId, folders)}>
                                                                    {isExpanded ? `${mentionResults.expandedChildren.length} file${mentionResults.expandedChildren.length !== 1 ? 's' : ''} • ` : ''}{folderAbsolutePath(file._folderId, folders)}
                                                                </p>
                                                            </div>
                                                        </button>
                                                        {isExpanded && (
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); tagAllInFolder(file._folderId); }}
                                                                className="w-full flex items-center justify-center gap-2 px-4 py-1.5 text-[10px] font-bold text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20 transition-all border-b border-white/5"
                                                            >
                                                                Tag all {mentionResults.expandedChildren.length} file{mentionResults.expandedChildren.length !== 1 ? 's' : ''}
                                                            </button>
                                                        )}
                                                        {isExpanded && mentionResults.expandedChildren.map((child) => {
                                                            const childFlatIdx = mentionResults.flatIndexMap[child.id];
                                                            return (
                                                            <button
                                                                key={child.id}
                                                                data-mention-idx={childFlatIdx}
                                                                onClick={() => selectMention(child)}
                                                                onMouseEnter={() => setMentionIndex(childFlatIdx)}
                                                                className={`w-full flex items-center gap-3 pl-10 pr-4 py-2 text-left transition-all ${childFlatIdx === mentionIndex
                                                                    ? 'bg-indigo-600/15 border-l-2 border-indigo-500'
                                                                    : 'hover:bg-white/5 border-l-2 border-transparent'
                                                                    }`}
                                                            >
                                                                <div className={`w-6 h-6 rounded flex items-center justify-center flex-shrink-0 ${childFlatIdx === mentionIndex ? 'bg-indigo-600/20 text-indigo-400' : 'bg-white/5 text-zinc-500'}`}>
                                                                    <Icons.File size={11} />
                                                                </div>
                                                                <div className="flex-1 min-w-0">
                                                                    <p className={`text-xs font-medium truncate ${childFlatIdx === mentionIndex ? 'text-white' : 'text-zinc-400'}`}>
                                                                        {child.filename}
                                                                    </p>
                                                                    <p className="text-[9px] text-zinc-600 truncate">
                                                                        {child.mime_type || 'file'}
                                                                    </p>
                                                                </div>
                                                                {isIndexSearchable(child) && (
                                                                    <div className="w-1.5 h-1.5 rounded-full bg-green-400 flex-shrink-0" title="Indexed"></div>
                                                                )}
                                                            </button>
                                                            );
                                                        })}
                                                        </React.Fragment>
                                                        );
                                                    })}
                                                    {mentionResults.folderCount > 0 && mentionResults.files.length > 0 && <div key="files-header" className="px-4 pt-2 pb-1 text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em]">Files</div>}
                                                    {mentionResults.files.map((file) => {
                                                        const flatIdx = mentionResults.flatIndexMap[file.id];
                                                        return (
                                                        <button
                                                            key={file.id}
                                                            data-mention-idx={flatIdx}
                                                            onClick={() => selectMention(file)}
                                                            onMouseEnter={() => setMentionIndex(flatIdx)}
                                                            className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-all ${flatIdx === mentionIndex
                                                                ? 'bg-indigo-600/15 border-l-2 border-indigo-500'
                                                                : 'hover:bg-white/5 border-l-2 border-transparent'
                                                                }`}
                                                        >
                                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${flatIdx === mentionIndex ? 'bg-indigo-600/20 text-indigo-400' : 'bg-white/5 text-zinc-500'}`}>
                                                                <Icons.File size={14} />
                                                            </div>
                                                            <div className="flex-1 min-w-0">
                                                                <p className={`text-sm font-medium truncate ${flatIdx === mentionIndex ? 'text-white' : 'text-zinc-300'}`}>
                                                                    {file._path && (
                                                                        <span className="text-zinc-600 font-normal">{file._path} / </span>
                                                                    )}
                                                                    {file.filename}
                                                                </p>
                                                                <p className="text-[10px] text-zinc-600 truncate">
                                                                    {file.mime_type || 'file'}
                                                                    {mentionTab === 'recent' && file.uploaded_at && (
                                                                        <span> • Added {new Date(file.uploaded_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>
                                                                    )}
                                                                </p>
                                                            </div>
                                                            {isIndexSearchable(file) && (
                                                                <div className="w-1.5 h-1.5 rounded-full bg-green-400 flex-shrink-0" title="Indexed"></div>
                                                            )}
                                                        </button>
                                                        );
                                                    })}
                                                </>
                                            )}
                                            {mentionTab === 'folders' && (
                                                <>
                                                    {mentionResults.allFolders.length === 0 && (
                                                        <div className="px-4 py-6 text-center">
                                                            <p className="text-xs text-zinc-500">No folders found</p>
                                                        </div>
                                                    )}
                                                    {mentionResults.allFolders.map((file) => {
                                                        const flatIdx = mentionResults.flatIndexMap[file.id];
                                                        const isExpanded = expandedFolderId === file._folderId;
                                                        return (
                                                        <React.Fragment key={file.id}>
                                                        <button
                                                            data-mention-idx={flatIdx}
                                                            onClick={() => selectMention(file)}
                                                            onMouseEnter={() => setMentionIndex(flatIdx)}
                                                            className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-all ${flatIdx === mentionIndex
                                                                ? 'bg-indigo-600/15 border-l-2 border-indigo-500'
                                                                : 'hover:bg-white/5 border-l-2 border-transparent'
                                                                }`}
                                                        >
                                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${flatIdx === mentionIndex ? 'bg-indigo-600/20 text-indigo-400' : 'bg-indigo-500/10 text-indigo-500'}`}>
                                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className={`transition-transform ${isExpanded ? 'rotate-90' : ''}`}><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                                                            </div>
                                                            <div className="min-w-0">
                                                                <p className={`text-sm font-medium truncate ${flatIdx === mentionIndex ? 'text-white' : 'text-zinc-300'}`}>
                                                                    {file.filename}
                                                                </p>
                                                                <p className="text-[10px] text-zinc-600 truncate" title={folderAbsolutePath(file._folderId, folders)}>
                                                                    {isExpanded ? `${mentionResults.expandedChildren.length} file${mentionResults.expandedChildren.length !== 1 ? 's' : ''} • ` : ''}{folderAbsolutePath(file._folderId, folders)}
                                                                </p>
                                                            </div>
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); selectMentionFolder(file); }}
                                                                title="Scope whole folder (live)"
                                                                className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded text-emerald-400 bg-emerald-500/15 hover:bg-emerald-500/30 transition-all flex-shrink-0 ml-1"
                                                            >
                                                                Tag
                                                            </button>
                                                        </button>
                                                        {isExpanded && (
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); tagAllInFolder(file._folderId); }}
                                                                className="w-full flex items-center justify-center gap-2 px-4 py-1.5 text-[10px] font-bold text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20 transition-all border-b border-white/5"
                                                            >
                                                                Tag all {mentionResults.expandedChildren.length} file{mentionResults.expandedChildren.length !== 1 ? 's' : ''}
                                                            </button>
                                                        )}
                                                        {isExpanded && mentionResults.expandedChildren.map((child) => {
                                                            const childFlatIdx = mentionResults.flatIndexMap[child.id];
                                                            return (
                                                            <button
                                                                key={child.id}
                                                                data-mention-idx={childFlatIdx}
                                                                onClick={() => selectMention(child)}
                                                                onMouseEnter={() => setMentionIndex(childFlatIdx)}
                                                                className={`w-full flex items-center gap-3 pl-10 pr-4 py-2 text-left transition-all ${childFlatIdx === mentionIndex
                                                                    ? 'bg-indigo-600/15 border-l-2 border-indigo-500'
                                                                    : 'hover:bg-white/5 border-l-2 border-transparent'
                                                                    }`}
                                                            >
                                                                <div className={`w-6 h-6 rounded flex items-center justify-center flex-shrink-0 ${childFlatIdx === mentionIndex ? 'bg-indigo-600/20 text-indigo-400' : 'bg-white/5 text-zinc-500'}`}>
                                                                    <Icons.File size={11} />
                                                                </div>
                                                                <div className="flex-1 min-w-0">
                                                                    <p className={`text-xs font-medium truncate ${childFlatIdx === mentionIndex ? 'text-white' : 'text-zinc-400'}`}>
                                                                        {child.filename}
                                                                    </p>
                                                                    <p className="text-[9px] text-zinc-600 truncate">
                                                                        {child.mime_type || 'file'}
                                                                    </p>
                                                                </div>
                                                                {isIndexSearchable(child) && (
                                                                    <div className="w-1.5 h-1.5 rounded-full bg-green-400 flex-shrink-0" title="Indexed"></div>
                                                                )}
                                                            </button>
                                                            );
                                                        })}
                                                        </React.Fragment>
                                                        );
                                                    })}
                                                </>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* No results message */}
                                {mentionActive && mentionResults.total === 0 && mentionQuery.length > 0 && (
                                    <div className="absolute bottom-full mb-2 left-0 right-0 bg-[#0c0c0d] border border-white/10 rounded-2xl shadow-2xl overflow-hidden z-50 animate-fade-in p-6 text-center">
                                        <Icons.Search size={20} className="text-zinc-600 mx-auto mb-2" />
                                        <p className="text-xs text-zinc-500 font-medium">No files or folders matching "{mentionQuery}" <span className="text-zinc-600">· Esc to dismiss, Enter to send anyway</span></p>
                                    </div>
                                )}

                                {/* Upload error pills */}
                                {toasts.length > 0 && (
                                    <div className="flex flex-col gap-1.5 mb-2">
                                        {toasts.map(t => (
                                            <div key={t.id} className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs w-fit max-w-full animate-fade-in ${t.type === 'info' ? 'bg-zinc-800/80 border-white/10 text-zinc-300' : 'bg-red-500/10 border-red-500/25 text-red-400'}`}>
                                                {t.type !== 'info' && (<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="shrink-0"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>)}
                                                <span className="truncate">{t.message}</span>
                                                {t.file && (
                                                    <button onClick={() => handleChatReplace(t)} className="shrink-0 flex items-center gap-1 px-2.5 py-1 rounded-full bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 active:scale-95 text-indigo-200 hover:text-white transition-all text-[10px] font-bold uppercase tracking-wide">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.75"/></svg>
                                                        Replace
                                                    </button>
                                                )}
                                                <button onClick={() => setToasts(prev => prev.filter(x => x.id !== t.id))} className="shrink-0 text-red-500/60 hover:text-red-400 transition-colors leading-none">✕</button>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div className="relative flex items-center bg-white/5 border border-white/10 rounded-[22px] p-1.5 shadow-2xl transition-all focus-within:border-white/20 focus-within:bg-white/[0.08]">
                                    {/* Add file — dropdown with 2 options */}
                                    <div className="relative flex-shrink-0 ml-1" ref={addMenuRef}>
                                        <button
                                            onClick={() => setShowAddMenu(v => !v)}
                                            className={`h-9 w-9 flex items-center justify-center rounded-xl transition-all ${showAddMenu ? 'text-indigo-400 bg-white/8' : 'text-zinc-500 hover:text-indigo-400 hover:bg-white/8'}`}
                                            title="Add file"
                                        >
                                            <Icons.Paperclip size={17} />
                                        </button>
                                        {showAddMenu && (
                                            <div className="absolute bottom-full mb-2 left-0 min-w-[200px] bg-[#18181b] border border-white/10 rounded-2xl shadow-2xl overflow-hidden z-50">
                                                <div className="p-1.5">
                                                    <button onClick={() => { setShowAddMenu(false); document.getElementById('chat-file-upload').click(); }}
                                                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-all text-left">
                                                        <Icons.Upload size={14} className="text-zinc-400 flex-shrink-0" />
                                                        <div>
                                                            <div className="text-[13px] font-semibold text-white">From computer</div>
                                                            <p className="text-[11px] text-zinc-500">Upload a new file</p>
                                                        </div>
                                                    </button>
                                                    <button onClick={() => { setShowAddMenu(false); setVaultPickerSelected(new Set(taggedFiles.map(f => f.id))); setVaultPickerSearch(''); setShowVaultPicker(true); }}
                                                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-all text-left">
                                                        <Icons.Home size={14} className="text-zinc-400 flex-shrink-0" />
                                                        <div>
                                                            <div className="text-[13px] font-semibold text-white">From vault</div>
                                                            <p className="text-[11px] text-zinc-500">Tag existing files</p>
                                                        </div>
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    <textarea
                                        data-testid="chat-input"
                                        ref={inputRef}
                                        rows={1}
                                        value={input}
                                        onChange={handleInputChange}
                                        onKeyDown={handleInputKeyDown}
                                        onPaste={handleChatPaste}
                                        placeholder={taggedFiles.length > 0 ? "Ask about your tagged files..." : "Ask anything... (@ to tag files)"}
                                        className="flex-1 bg-transparent border-none px-4 py-3 text-white focus:ring-0 outline-none focus:outline-none placeholder:text-zinc-600 text-sm resize-none"
                                        style={{ border: 'none', boxShadow: 'none', maxHeight: '160px', overflowY: 'auto' }}
                                    />
                                    {/* Hidden file input */}
                                    <input
                                        type="file"
                                        multiple
                                        className="hidden"
                                        id="chat-file-upload"
                                        onChange={async (e) => {
                                            const files = Array.from(e.target.files || []);
                                            if (!files.length) return;
                                            e.target.value = '';
                                            for (const file of files) {
                                                try {
                                                    const uploaded = await api.uploadFile(file);
                                                    const fid = uploaded.file_id;
                                                    setTaggedFiles(prev => [...prev, { id: fid, filename: uploaded.filename }]);
                                                    try {
                                                        await api.grantAIAccess(fid);
                                                        chatUploadedIdsRef.current.add(fid);
                                                        setIndexingTagIds(prev => new Set([...prev, fid]));
                                                    } catch (grantError) {
                                                        showToast(describeIndexingError(grantError, file.name), 'info');
                                                    }
                                                    refresh && refresh(); // update files list so @ mention sees the new file
                                                } catch(err) {
                                                    const isDuplicate = err.data && err.data.existing_id;
                                                    if (isDuplicate) {
                                                        const eid = err.data.existing_id;
                                                        setTaggedFiles(prev => {
                                                            const alreadyTagged = prev.some(t => t.id === eid);
                                                            if (alreadyTagged) return prev;
                                                            return [...prev, { id: eid, filename: err.data.filename }];
                                                        });
                                                        showToast(`"${file.name}" already exists — tagged existing file.`, 'error', file);
                                                    } else {
                                                        showToast(`${file.name}: ${err.message}`, 'error', file);
                                                    }
                                                }
                                            }
                                        }}
                                    />
                                    {/* Vault file picker modal */}
                                    {showVaultPicker && (
                                        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[300] flex items-center justify-center" onClick={() => setShowVaultPicker(false)}>
                                            <div className="bg-[#111113] border border-white/10 rounded-2xl w-full max-w-md mx-4 shadow-2xl flex flex-col max-h-[60vh]" onClick={e => e.stopPropagation()}>
                                                <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between flex-shrink-0">
                                                    <h3 className="text-white font-semibold text-sm">Add from Vault</h3>
                                                    <button onClick={() => setShowVaultPicker(false)} className="text-zinc-500 hover:text-white transition-colors"><Icons.X size={16} /></button>
                                                </div>
                                                <div className="px-4 py-3 border-b border-white/5 flex-shrink-0">
                                                    <input autoFocus type="text" value={vaultPickerSearch} onChange={e => setVaultPickerSearch(e.target.value)}
                                                        placeholder="Search files…"
                                                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm text-white placeholder:text-zinc-600 outline-none focus:border-indigo-500/50" />
                                                </div>
                                                <div className="flex-1 overflow-y-auto p-2">
                                                    {files.filter(f => !f.is_deleted && (f.original_filename || f.filename || '').toLowerCase().includes(vaultPickerSearch.toLowerCase())).map(f => {
                                                        const fid = f.id;
                                                        const fname = f.original_filename || f.filename || `File #${fid}`;
                                                        const isChecked = vaultPickerSelected.has(fid);
                                                        return (
                                                            <button key={fid} onClick={() => setVaultPickerSelected(prev => { const n = new Set(prev); if (n.has(fid)) n.delete(fid); else n.add(fid); return n; })}
                                                                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left ${isChecked ? 'bg-indigo-600/15 border border-indigo-500/20' : 'hover:bg-white/5 border border-transparent'}`}>
                                                                <div className={`w-4 h-4 rounded flex-shrink-0 border-2 flex items-center justify-center transition-all ${isChecked ? 'bg-indigo-600 border-indigo-500' : 'border-zinc-600'}`}>
                                                                    {isChecked && <svg width="8" height="8" viewBox="0 0 10 10" fill="none"><path d="M2 5L4.5 7.5L8 3" stroke="white" strokeWidth="1.5" strokeLinecap="round"/></svg>}
                                                                </div>
                                                                <div className="flex-1 min-w-0">
                                                                    <div className="text-sm text-white truncate">{fname}</div>
                                                                    {isIndexSearchable(f) && <div className="text-[10px] text-indigo-400">Indexed</div>}
                                                                </div>
                                                            </button>
                                                        );
                                                    })}
                                                    {files.filter(f => !f.is_deleted).length === 0 && <div className="text-center py-8 text-zinc-600 text-sm">No files in vault</div>}
                                                </div>
                                                <div className="px-4 py-3 border-t border-white/5 flex items-center justify-between flex-shrink-0">
                                                    <span className="text-xs text-zinc-500">{vaultPickerSelected.size} selected</span>
                                                    <button onClick={() => {
                                                        const picked = files.filter(f => vaultPickerSelected.has(f.id));
                                                        const blocked = picked.filter(f => classifyTagTarget(f).verdict === 'unsupported');
                                                        if (blocked.length > 0) {
                                                            showToast(`Skipped ${blocked.length} unreadable file${blocked.length === 1 ? '' : 's'} (zip/video can't be indexed).`, 'error', null, 5000);
                                                        }
                                                        const newTagged = picked
                                                            .filter(f => classifyTagTarget(f).verdict !== 'unsupported')
                                                            .map(f => {
                                                                const verdict = classifyTagTarget(f);
                                                                return {
                                                                    id: f.id,
                                                                    filename: f.original_filename || f.filename,
                                                                    ...(verdict.verdict === 'unindexed' ? { _warn: verdict.reason } : {}),
                                                                };
                                                            });
                                                        setTaggedFiles(newTagged);
                                                        setShowVaultPicker(false);
                                                    }} className="px-4 py-2 bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white text-sm rounded-xl transition-all">
                                                        Add Selected
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                    {/* PROMPT selector — Grok style dropdown */}
                                    <div className="relative flex-shrink-0 mr-1" ref={promptDropdownRef}>
                                        <button
                                            onClick={() => setShowPromptDropdown(v => !v)}
                                            className="flex items-center gap-1.5 pl-3 pr-2 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border-0 transition-all duration-200"
                                            title="Switch prompt style"
                                        >
                                            <span className="text-[11px] font-bold text-white/90 tracking-tight">
                                                {PROMPT_OPTIONS.find(p => p.key === activePrompt)?.label || 'Casual'}
                                            </span>
                                            <Icons.ChevronDown size={12} className={`text-zinc-400 transition-transform duration-200 ${showPromptDropdown ? 'rotate-180' : ''}`} />
                                        </button>

                                        {showPromptDropdown && (
                                            <div className="absolute bottom-full mb-2 left-0 min-w-[220px] bg-[#18181b] border border-white/10 rounded-2xl shadow-2xl overflow-hidden z-50 animate-fade-in">
                                                <div className="p-1.5">
                                                    {PROMPT_OPTIONS.map(opt => (
                                                        <button
                                                            key={opt.key}
                                                            onClick={() => {
                                                                setActivePrompt(opt.key);
                                                                localStorage.setItem('activePrompt', opt.key);
                                                                setShowPromptDropdown(false);
                                                            }}
                                                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left group ${activePrompt === opt.key ? 'bg-white/8' : 'hover:bg-white/5'}`}
                                                        >
                                                            <span className="text-base w-5 flex-shrink-0">{opt.icon}</span>
                                                            <div className="flex-1 min-w-0">
                                                                <div className="flex items-center gap-1.5">
                                                                    <span className="text-[13px] font-semibold text-white">{opt.label}</span>
                                                                </div>
                                                                <p className="text-[11px] text-zinc-500 mt-0.5">{opt.desc}</p>
                                                            </div>
                                                            {activePrompt === opt.key && (
                                                                <Icons.Check size={14} className="text-white flex-shrink-0" />
                                                            )}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    {/* Send/Stop button — dynamic based on processing state */}
                                    {processing ? (
                                        <button
                                            onClick={(e) => { e.preventDefault(); cancelRequest(); }}
                                            className="h-9 w-9 flex-shrink-0 rounded-full bg-red-600/20 hover:bg-red-500/40 border border-red-500/30 text-red-500 hover:text-red-400 flex items-center justify-center transition-all active:scale-95 z-50 relative pointer-events-auto"
                                            title="Stop Generation"
                                        >
                                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg>
                                        </button>
                                    ) : (
                                        <button
                                            data-testid="chat-send"
                                            onClick={() => sendWithTags()}
                                            disabled={!input.trim() || indexingTagIds.size > 0 || processing}
                                            className="h-9 w-9 flex-shrink-0 rounded-full bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white flex items-center justify-center transition-all active:scale-95 disabled:opacity-20 disabled:grayscale disabled:scale-100 disabled:cursor-not-allowed"
                                            title={indexingTagIds.size > 0 ? 'Waiting for file to be indexed…' : 'Send'}
                                        >
                                            {indexingTagIds.size > 0 ? (
                                                <Icons.Loader size={14} className="animate-spin" />
                                            ) : (
                                                <Icons.ArrowUp size={16} />
                                            )}
                                        </button>
                                    )}
                                </div>
                            </div>

                            {/* Bottom row: toggles left, footer right */}
                            <div className="flex items-center justify-between mt-2 px-1">
                                {/* Left: DEEP + WEB toggles */}
                                <div className="flex items-center gap-2">
                                    {/* FILE SEARCH toggle (PostgreSQL vectors) */}
                                    <button
                                        data-testid="chat-files-toggle"
                                        aria-pressed={fileSearch}
                                        onClick={toggleFileSearch}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all duration-200 ${fileSearch ? 'bg-emerald-600/10 border-emerald-500/30' : 'bg-white/4 border-white/10 hover:border-white/20'}`}
                                        title={fileSearch ? "Vault Search ON — query indexed files" : "Vault Search OFF — enable file retrieval"}
                                    >
                                        <Icons.Database size={11} className={fileSearch ? 'text-emerald-400' : 'text-zinc-600'} />
                                        <span className={`text-[10px] font-bold uppercase tracking-wider ${fileSearch ? 'text-emerald-400' : 'text-zinc-600'}`}>Vault</span>
                                        <div className={`relative w-7 h-3.5 rounded-full transition-all duration-200 ${fileSearch ? 'bg-emerald-600' : 'bg-white/15'}`}>
                                            <div className={`absolute top-[1.5px] left-[2px] w-2.5 h-2.5 bg-white rounded-full shadow transition-transform duration-200 ${fileSearch ? 'translate-x-3' : 'translate-x-0'}`}></div>
                                        </div>
                                    </button>

                                    {/* WEB toggle */}
                                    <button
                                        data-testid="chat-web-toggle"
                                        aria-pressed={webSearch}
                                        onClick={toggleWebSearch}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all duration-200 ${webSearch ? 'bg-emerald-600/10 border-emerald-500/30' : 'bg-white/4 border-white/10 hover:border-white/20'}`}
                                        title={webSearch ? "Web Search ON — live internet results" : "Web Search OFF — enable web search"}
                                    >
                                        <div className="relative flex-shrink-0">
                                            <Icons.Globe size={11} className={webSearch ? 'text-emerald-400' : 'text-zinc-600'} />
                                            {webSearch && <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span>}
                                        </div>
                                        <span className={`text-[10px] font-bold uppercase tracking-wider ${webSearch ? 'text-emerald-400' : 'text-zinc-600'}`}>Web</span>
                                        <div className={`relative w-7 h-3.5 rounded-full transition-all duration-200 ${webSearch ? 'bg-emerald-600' : 'bg-white/15'}`}>
                                            <div className={`absolute top-[1.5px] left-[2px] w-2.5 h-2.5 bg-white rounded-full shadow transition-transform duration-200 ${webSearch ? 'translate-x-3' : 'translate-x-0'}`}></div>
                                        </div>
                                    </button>
                                </div>

                                {/* Right: footer */}
                                <div className="flex items-center gap-2 opacity-40">
                                    <Icons.Shield size={10} className="text-zinc-500" />
                                    <p className="text-[9px] text-zinc-500 font-bold uppercase tracking-[0.2em]">
                                        Powered by Lavix AI
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Prompt A / B Editor Modal ── */}
                {showPersonaEditor && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in" onClick={() => setShowPersonaEditor(false)}>
                        <div className="bg-[#111113] border border-white/10 rounded-2xl shadow-2xl w-full max-w-lg mx-4 p-6" onClick={e => e.stopPropagation()}>
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-2">
                                    <Icons.Sparkles size={18} className="text-indigo-400" />
                                    <h3 className="text-sm font-bold text-white">AI Persona Prompts</h3>
                                </div>
                                <button onClick={() => setShowPersonaEditor(false)} className="text-zinc-500 hover:text-white transition-colors"><Icons.X size={18} /></button>
                            </div>
                            {/* Tab selector */}
                            <div className="flex gap-2 mb-4">
                                {['A', 'B'].map(tab => (
                                    <button key={tab} onClick={() => setEditingTab(tab)}
                                        className={`flex-1 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${editingTab === tab ? (tab === 'A' ? 'bg-indigo-600 text-white' : 'bg-amber-500 text-white') : 'bg-white/5 text-zinc-500 hover:bg-white/10'}`}>
                                        Prompt {tab} {activePrompt === tab ? '(active)' : ''}
                                    </button>
                                ))}
                            </div>
                            <p className="text-xs text-zinc-500 mb-3">
                                {editingTab === 'A' ? 'Casual style. Toggle "Prompt A" to activate.' : 'Professional / structured style. Toggle "Prompt B" to activate.'}
                            </p>
                            <textarea
                                value={editingTab === 'A' ? personaPromptA : personaPromptB}
                                onChange={e => editingTab === 'A' ? setPersonaPromptA(e.target.value) : setPersonaPromptB(e.target.value)}
                                className="w-full h-48 bg-black/30 border border-white/10 rounded-xl p-3 text-xs text-zinc-200 font-mono resize-none focus:outline-none focus:border-indigo-500/50"
                            />
                            <div className="flex items-center justify-between mt-4">
                                <button onClick={() => resetPrompt(editingTab)} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/5">
                                    Reset {editingTab} to default
                                </button>
                                <div className="flex items-center gap-2">
                                    <button onClick={() => setShowPersonaEditor(false)} className="text-xs text-zinc-400 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors">Cancel</button>
                                    <button onClick={savePersonaPrompts} className="text-xs font-bold px-4 py-1.5 rounded-lg bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white transition-colors">
                                        Save
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Floating text selection menu */}
                {floatingMenu && (
                    <div
                        className="floating-menu flex gap-1 bg-zinc-900 border border-white/10 rounded-lg px-1.5 py-1 shadow-xl"
                        style={{ left: floatingMenu.x, top: floatingMenu.y }}
                    >
                        <button
                            onClick={async () => { const text = floatingMenu.text; setFloatingMenu(null); try { if (navigator.clipboard?.writeText) { await navigator.clipboard.writeText(text); } else { throw new Error(''); } } catch(e) { const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta); } }}
                            className="text-[11px] text-zinc-300 hover:text-white px-2 py-0.5 rounded hover:bg-white/10 flex items-center gap-1 transition-all"
                        >
                            <Icons.Copy size={10}/> Copy
                        </button>
                        <button
                            onClick={() => {
                                const q = `Explain: "${floatingMenu.text.slice(0, 100)}"`;
                                setFloatingMenu(null);
                                sendFollowup(q);
                            }}
                            className="text-[11px] text-zinc-300 hover:text-white px-2 py-0.5 rounded hover:bg-white/10 flex items-center gap-1 transition-all"
                        >
                            <Icons.Search size={10}/> Ask
                        </button>
                    </div>
                )}
                </React.Fragment>
            );
        }


        // ── GLOBAL UPLOAD WIDGET ─────────────────────────────────────────────────
        function GlobalUploadWidget({ uploading, progress, currentFileIndex, totalFiles, onMaximize }) {
            if (!uploading) return null;

            return (
                <div
                    onClick={onMaximize}
                    className="fixed bottom-6 right-6 z-50 bg-zinc-900 border border-white/10 rounded-2xl p-4 shadow-2xl cursor-pointer hover:bg-zinc-800 transition-all group flex items-center gap-4 max-w-sm animate-fade-in"
                >
                    <div className="relative w-10 h-10 flex-shrink-0">
                        <svg className="w-full h-full transform -rotate-90">
                            <circle className="text-white/5" strokeWidth="3" stroke="currentColor" fill="transparent" r="18" cx="20" cy="20" />
                            <circle
                                className="text-indigo-500 transition-all duration-300"
                                strokeWidth="3"
                                strokeDasharray={2 * Math.PI * 18}
                                strokeDashoffset={(2 * Math.PI * 18) * (1 - progress / 100)}
                                strokeLinecap="round"
                                stroke="currentColor"
                                fill="transparent"
                                r="18" cx="20" cy="20"
                            />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-[10px] font-bold text-white">{Math.round(progress)}%</span>
                        </div>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-xs font-bold text-white mb-0.5">Encrypting Files</p>
                        <p className="text-[10px] text-zinc-400 truncate">File {currentFileIndex + 1} of {totalFiles}</p>
                    </div>
                    <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                        <Icons.Maximize size={14} />
                    </div>
                </div>
            );
        }


        // ── UPLOAD VIEW ──────────────────────────────────────────────────────────
        function UploadView({
            files,
            uploading,
            progress,
            uploadResults,
            showResults,
            currentFileIndex,
            estimatedRemaining,
            handleFiles,
            handleDismissResults,
            onMinimize,
            onPreview,
            onReplace,
            folders,
            uploadTargetFolder,
            setUploadTargetFolder,
            onCreateFolder,
            onFoldersRefresh,
            handleFilePairs
        }) {
            const [dragActive, setDragActive] = useState(false);
            const [newFolderMode, setNewFolderMode] = React.useState(false);
            const [newFolderName, setNewFolderName] = React.useState('');
            const [creatingFolder, setCreatingFolder] = React.useState(false);

            const handleCreateFolder = async () => {
                const name = newFolderName.trim();
                if (!name) return;
                setCreatingFolder(true);
                try {
                    await api.createFolder(name, null);
                    setNewFolderName('');
                    setNewFolderMode(false);
                    onFoldersRefresh && await onFoldersRefresh();
                } catch(e) {}
                setCreatingFolder(false);
            };



            const handleDrop = async (e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragActive(false);

                const items = e.dataTransfer.items;
                if (items && items.length > 0 && items[0].webkitGetAsEntry) {
                    // Recursively mirror a directory entry into vault, creating sub-folders as needed.
                    // Returns array of { file, folderId } pairs.
                    const mirrorDir = async (entry, parentVaultId) => {
                        const pairs = [];
                        const readChildren = (dirEntry) => new Promise(resolve => {
                            const reader = dirEntry.createReader();
                            const all = [];
                            const read = () => reader.readEntries(async batch => {
                                if (!batch.length) { resolve(all); return; }
                                all.push(...batch);
                                read();
                            }, () => resolve(all));
                            read();
                        });

                        const children = await readChildren(entry);
                        await Promise.all(children.map(async child => {
                            if (child.isFile) {
                                await new Promise(res => child.file(f => { pairs.push({ file: f, folderId: parentVaultId }); res(); }, res));
                            } else if (child.isDirectory) {
                                try {
                                    const sub = await api.createFolder(child.name, parentVaultId);
                                    onFoldersRefresh && onFoldersRefresh();
                                    const nested = await mirrorDir(child, sub.id);
                                    pairs.push(...nested);
                                } catch(_) {
                                    const nested = await mirrorDir(child, parentVaultId);
                                    pairs.push(...nested);
                                }
                            }
                        }));
                        return pairs;
                    };

                    const entryList = [...items].map(i => i.webkitGetAsEntry()).filter(Boolean);

                    if (entryList.some(e => e.isDirectory)) {
                        const allPairs = [];
                        for (const entry of entryList) {
                            if (entry.isDirectory) {
                                try {
                                    const topFolder = await api.createFolder(entry.name, uploadTargetFolder);
                                    onFoldersRefresh && onFoldersRefresh();
                                    // mirrorDir already collects ALL files (top-level + all sub-dirs)
                                    const pairs = await mirrorDir(entry, topFolder.id);
                                    allPairs.push(...pairs);
                                } catch(err) {
                                    console.error('Failed to create top-level folder:', err);
                                }
                            } else if (entry.isFile) {
                                await new Promise(res => entry.file(f => { allPairs.push({ file: f, folderId: uploadTargetFolder }); res(); }, res));
                            }
                        }
                        // Single upload pass — no concurrent races
                        if (allPairs.length) handleFilePairs(allPairs);
                        return;
                    }

                    // All files — upload flat
                    const allFiles = [];
                    await Promise.all(entryList.map(entry => new Promise(res => {
                        if (entry.isFile) entry.file(f => { allFiles.push(f); res(); }, res);
                        else res();
                    })));
                    if (allFiles.length) { handleFiles(allFiles); return; }
                }

                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    handleFiles(e.dataTransfer.files);
                }
            };

            // Show results screen
            if (showResults && uploadResults.length > 0) {
                const successCount = uploadResults.filter(r => r.success).length;
                const failedCount = uploadResults.filter(r => !r.success).length;

                return (
                    <div className="flex-1 p-8 bg-black flex flex-col items-center pt-16 animate-fade-in h-[calc(100vh-64px)] overflow-auto">
                        <div className="w-full max-w-2xl">
                            <h2 className="text-xl font-bold text-white mb-6">Upload Results</h2>

                            <div className="glass-card glass-card-static p-6 mb-6">
                                <div className="flex items-center gap-4 mb-4">
                                    {failedCount > 0 ? (
                                        <div className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center">
                                            <Icons.AlertTriangle size={24} className="text-amber-400" />
                                        </div>
                                    ) : (
                                        <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                                            <Icons.Check size={24} className="text-green-400" />
                                        </div>
                                    )}
                                    <div>
                                        <p className="text-white font-medium">
                                            {successCount} of {uploadResults.length} files uploaded successfully
                                        </p>
                                        {failedCount > 0 && (
                                            <p className="text-red-400 text-sm">{failedCount} file(s) failed</p>
                                        )}
                                    </div>
                                </div>

                                <div className="space-y-2 max-h-64 overflow-y-auto">
                                    {uploadResults.map((result, i) => (
                                        <div key={i} className={`flex items-center gap-3 p-3 rounded-lg ${result.success ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                                            {result.success ? (
                                                <Icons.Check size={16} className="text-green-400 flex-shrink-0" />
                                            ) : (
                                                <Icons.X size={16} className="text-red-400 flex-shrink-0" />
                                            )}
                                            {!result.success && result.existing_id ? (
                                                <button
                                                    onClick={() => onPreview({ id: result.existing_id, filename: result.name, mime_type: result.existing_mime })}
                                                    className="text-sm line-clamp-2 break-all flex-1 text-left hover:underline text-red-300 transition-all font-medium"
                                                >
                                                    {result.name}
                                                </button>
                                            ) : (
                                                <span className={`text-sm line-clamp-2 break-all flex-1 ${result.success ? 'text-green-300' : 'text-red-300'}`}>
                                                    {result.name}
                                                </span>
                                            )}
                                            {!result.success && (
                                                <div className="flex items-center gap-3">
                                                    <span className="text-[10px] text-red-400/80 italic font-medium whitespace-nowrap">{result.error}</span>
                                                    {result.existing_id && (
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onReplace(result.file_obj); }}
                                                            className="px-3 py-1 bg-red-500/20 hover:bg-red-500/40 text-red-400 text-[10px] font-black uppercase rounded-md border border-red-500/30 transition-all hover:scale-105 active:scale-95 flex-shrink-0"
                                                        >
                                                            Replace
                                                        </button>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={handleDismissResults}
                                className="w-full bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-100 border border-indigo-500/20 py-3 rounded-xl font-medium transition-all shadow-[0_0_15px_rgba(99,102,241,0.1)]"
                            >
                                Continue to Files
                            </button>
                        </div>
                    </div>
                );
            }

            if (uploading) {
                const radius = 54;
                const circumference = 2 * Math.PI * radius;
                const offset = circumference - (progress / 100) * circumference;

                return (
                    <div className="flex-1 p-8 bg-black flex flex-col items-center pt-24 animate-fade-in h-[calc(100vh-64px)] overflow-hidden">
                        <div className="w-full max-w-2xl flex flex-col items-center">
                            <h2 className="text-[10px] font-black text-zinc-600 mb-12 tracking-[0.3em] uppercase">Encryption Tunnel Active</h2>

                            <div className="relative w-48 h-48 flex items-center justify-center mb-10">
                                {/* Outer Glow Ring */}
                                <div className="absolute inset-0 rounded-full border border-indigo-500/10 scale-110 blur-sm"></div>
                                <div className="absolute inset-0 rounded-full border border-indigo-500/5 scale-125 blur-md"></div>

                                <svg className="w-full h-full transform -rotate-90">
                                    <circle
                                        className="text-white/5"
                                        strokeWidth="2"
                                        stroke="currentColor"
                                        fill="transparent"
                                        r={radius}
                                        cx="96" cy="96"
                                    />
                                    <circle
                                        className="text-indigo-500 transition-all duration-300 ease-out"
                                        strokeWidth="4"
                                        strokeDasharray={circumference}
                                        strokeDashoffset={offset}
                                        strokeLinecap="round"
                                        stroke="currentColor"
                                        fill="transparent"
                                        r={radius}
                                        cx="96" cy="96"
                                        style={{ filter: 'drop-shadow(0 0 8px rgba(99, 102, 241, 0.5))' }}
                                    />
                                </svg>
                                <div className="absolute flex flex-col items-center">
                                    <span className="text-4xl font-black text-white tracking-tighter">{Math.round(progress)}%</span>
                                    <span className="text-[9px] font-bold text-indigo-400 mt-1 uppercase tracking-widest">Encrypting</span>
                                </div>
                            </div>

                            <div className="text-center space-y-3">
                                <p className="text-xl font-bold text-white tracking-tight">
                                    Encrypting <span className="text-indigo-400">{files[currentFileIndex]?.name}</span>
                                </p>
                                <div className="flex items-center justify-center gap-3">
                                    <div className="flex gap-1">
                                        {[...Array(3)].map((_, i) => (
                                            <div key={i} className="w-1.5 h-1.5 rounded-full bg-indigo-500/20 animate-pulse" style={{ animationDelay: `${i * 0.2}s` }}></div>
                                        ))}
                                    </div>
                                    <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest">
                                        File {currentFileIndex + 1} / {files.length}
                                    </p>
                                </div>
                            </div>

                            <div className="mt-12 px-6 py-3 bg-white/[0.02] rounded-2xl border border-white/5 flex items-center gap-3 backdrop-blur-xl">
                                <div className="text-red-500">
                                    <Icons.Clock size={14} />
                                </div>
                                <span className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.15em]">
                                    Est. {estimatedRemaining()} remaining
                                </span>
                            </div>

                            <button
                                onClick={onMinimize}
                                className="mt-8 px-8 py-3 bg-white/[0.03] hover:bg-white/[0.05] rounded-xl border border-white/5 hover:border-indigo-500/30 flex items-center gap-3 transition-all group"
                            >
                                <div className="w-5 h-5 rounded-full border border-indigo-500/30 flex items-center justify-center group-hover:border-indigo-500 transition-colors">
                                    <Icons.Loader className="w-3 h-3 text-indigo-500" />
                                </div>
                                <span className="text-xs font-bold text-zinc-400 group-hover:text-white uppercase tracking-wider">
                                    Minimize to Background
                                </span>
                            </button>
                        </div>
                    </div>
                );
            }

            return (
                <div className="flex-1 p-8 bg-[#000000] flex flex-col items-center pt-24 animate-fade-in relative h-screen overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/5 to-transparent pointer-events-none"></div>

                    <div
                        onDragEnter={() => setDragActive(true)}
                        onDragLeave={() => setDragActive(false)}
                        onDragOver={e => e.preventDefault()}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById('file-upload').click()}
                        className={`w-full max-w-2xl min-h-[320px] rounded-[32px] border-2 border-dashed flex flex-col items-center justify-center transition-all relative z-10 cursor-pointer ${dragActive
                            ? 'border-indigo-500 bg-indigo-500/10 shadow-[0_0_50px_rgba(99,102,241,0.2)] scale-[1.01]'
                            : 'border-white/5 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]'
                            }`}
                    >
                        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-4 transition-all ${dragActive ? 'bg-indigo-600 text-white shadow-lg' : 'bg-black border border-white/5 text-zinc-600'
                            }`}>
                            <Icons.Upload size={24} />
                        </div>
                        <h3 className="text-lg font-bold text-white mb-1 tracking-tight">Drop Files or Folders Here</h3>
                        <p className="text-zinc-600 text-xs font-medium mb-5">or choose what to upload</p>

                        {/* Upload buttons — stop propagation so they don't open the file picker */}
                        <div className="flex items-center gap-3" onClick={e => e.stopPropagation()}>
                            <button
                                onClick={() => document.getElementById('file-upload').click()}
                                className="flex items-center gap-2 px-4 py-2 bg-indigo-600/20 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-600/30 rounded-xl text-xs font-bold transition-all">
                                <Icons.Upload size={14} /> Upload Files
                            </button>
                            <button
                                onClick={() => document.getElementById('folder-upload').click()}
                                className="flex items-center gap-2 px-4 py-2 bg-white/[0.04] border border-white/[0.10] text-zinc-300 hover:bg-white/[0.08] hover:text-white rounded-xl text-xs font-bold transition-all">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400">
                                    <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
                                </svg>
                                Upload Folder
                            </button>
                        </div>

                        <input type="file" multiple className="hidden" id="file-upload"
                            onChange={e => handleFiles(e.target.files)} />
                        <input type="file" className="hidden" id="folder-upload"
                            webkitdirectory="" mozdirectory="" directory=""
                            onChange={async e => {
                                const fileList = [...e.target.files];
                                if (!fileList.length) return;
                                // Use webkitRelativePath to mirror folder structure into vault
                                // e.g. "MyFolder/SubA/doc.pdf" → create MyFolder, SubA inside it, upload doc.pdf there
                                const folderCache = {}; // "path" → vaultFolderId
                                const getOrCreateFolder = async (parts, parentId) => {
                                    const key = parts.join('/');
                                    if (folderCache[key] !== undefined) return folderCache[key];
                                    try {
                                        const f = await api.createFolder(parts[parts.length - 1], parentId);
                                        onFoldersRefresh && onFoldersRefresh();
                                        folderCache[key] = f.id;
                                        return f.id;
                                    } catch(_) { folderCache[key] = parentId; return parentId; }
                                };
                                // Build {file, folderId} pairs for a single upload pass
                                const pairs = [];
                                for (const file of fileList) {
                                    const parts = file.webkitRelativePath.split('/');
                                    // parts[0] = root folder name, parts[1..n-1] = sub-dirs, parts[n] = filename
                                    let parentId = uploadTargetFolder;
                                    for (let i = 0; i < parts.length - 1; i++) {
                                        const pathSoFar = parts.slice(0, i + 1);
                                        parentId = await getOrCreateFolder(pathSoFar, i === 0 ? uploadTargetFolder : folderCache[parts.slice(0, i).join('/')]);
                                    }
                                    pairs.push({ file, folderId: parentId });
                                }
                                if (pairs.length) handleFilePairs(pairs);
                                e.target.value = '';
                            }} />
                    </div>

                    {/* Folder target selector + New Folder */}
                    <div className="w-full max-w-2xl mt-4 space-y-2">
                        <div className="flex items-center gap-3 px-4 py-3 bg-white/[0.03] border border-white/[0.07] rounded-2xl">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400 flex-shrink-0">
                                <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
                            </svg>
                            <span className="text-xs font-semibold text-zinc-400 flex-shrink-0">Upload to</span>
                            <div className="flex-1 min-w-0">
                            <VaultDropdown
                                value={uploadTargetFolder ?? ''}
                                ariaLabel="Upload target folder"
                                title={`Upload to ${folderAbsolutePath(uploadTargetFolder ?? null, folders)}`}
                                options={[{ value: '', label: '/Root' }, ...(folders || []).map(f => ({ value: String(f.id), label: folderAbsolutePath(f.id, folders) }))]}
                                onChange={(v) => setUploadTargetFolder && setUploadTargetFolder(v ? Number(v) : null)}
                            />
                            </div>
                            <button
                                onClick={() => { setNewFolderMode(v => !v); setNewFolderName(''); }}
                                className="flex-shrink-0 flex items-center gap-1 px-2.5 py-1.5 bg-indigo-600/15 border border-indigo-500/25 text-indigo-300 hover:bg-indigo-600/25 rounded-xl text-xs font-bold transition-all whitespace-nowrap">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 5v14M5 12h14"/></svg>
                                New Folder
                            </button>
                        </div>
                        {newFolderMode && (
                            <div className="flex items-center gap-2 px-4 py-3 bg-indigo-600/[0.06] border border-indigo-500/20 rounded-2xl animate-fade-in">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className="text-indigo-400 flex-shrink-0">
                                    <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
                                </svg>
                                <input
                                    autoFocus
                                    value={newFolderName}
                                    onChange={e => setNewFolderName(e.target.value)}
                                    onKeyDown={e => { if (e.key === 'Enter') handleCreateFolder(); if (e.key === 'Escape') setNewFolderMode(false); }}
                                    placeholder="Folder name..."
                                    className="flex-1 bg-transparent text-xs text-zinc-200 outline-none placeholder-zinc-600 min-w-0"
                                />
                                <button onClick={handleCreateFolder} disabled={!newFolderName.trim() || creatingFolder}
                                    className="px-3 py-1 bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 rounded-lg text-xs font-bold transition-all hover:bg-indigo-600/50 disabled:opacity-40">
                                    {creatingFolder ? '...' : 'Create'}
                                </button>
                                <button onClick={() => setNewFolderMode(false)} className="p-1 text-zinc-600 hover:text-zinc-300 transition-colors">
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                                </button>
                            </div>
                        )}
                    </div>

                    {files.length > 0 && !uploading && (
                        <div className="w-full max-w-2xl mt-8 text-zinc-600 text-[10px] font-black uppercase tracking-[0.2em] animate-pulse text-center">
                            Initializing Pipeline...
                        </div>
                    )}
                </div>
            );
        }


        // ── SETTINGS VIEW ────────────────────────────────────────────────────────
        const GRAPH_MEMORY_PAGE_SIZE = 3;

        function SettingsView({ userProfile, onProfileUpdate, onUsernameChange, initialTab = 'profile', prefFontSize, setPrefFontSize, prefTimestamps, setPrefTimestamps, prefSendOnEnter, setPrefSendOnEnter, prefAutoScroll, setPrefAutoScroll, onConfirmAction }) {
            const [tab, setTab] = useState(initialTab);
            // AI Models lives under ADMIN: never strand non-admins on it.
            React.useEffect(() => {
                if ((tab === 'api_keys' || tab === 'admin_preferences') && !userProfile?.is_admin) setTab('profile');
            }, [tab, userProfile?.is_admin]);
            const [persona, setPersona] = useState(userProfile?.persona_prompt || '');
            const [saving, setSaving] = useState(false);
            const [saveMsg, setSaveMsg] = useState('');
            const [editUsername, setEditUsername] = useState(userProfile?.username || '');
            const [usernameSaving, setUsernameSaving] = useState(false);
            const [usernameMsg, setUsernameMsg] = useState('');
            const [currentPw, setCurrentPw] = useState('');
            const [newPw, setNewPw] = useState('');
            const [confirmPw, setConfirmPw] = useState('');
            const [pwMsg, setPwMsg] = useState('');
            const [sessions, setSessions] = useState([]);
            const [sessionsLoading, setSessionsLoading] = useState(false);
            const [summaryCopied, setSummaryCopied] = useState(false);
            const summaryCopiedTimer = React.useRef(null);
            React.useEffect(() => () => {
                if (summaryCopiedTimer.current) clearTimeout(summaryCopiedTimer.current);
            }, []);
            const [memoryEnabled, setMemoryEnabled] = useState(false);
            const [memoryItems, setMemoryItems] = useState([]);
            const [graphMemory, setGraphMemory] = useState(null);
            const [memoryConsentMismatch, setMemoryConsentMismatch] = useState(false);
            const [graphMemoryItems, setGraphMemoryItems] = useState([]);
            const [graphMemoryPage, setGraphMemoryPage] = useState(0);
            const [editingGraphMemoryId, setEditingGraphMemoryId] = useState(null);
            const [graphMemoryDraft, setGraphMemoryDraft] = useState({ subject: '', predicate: '', objectValue: '' });
            const [highlightedGraphMemoryId, setHighlightedGraphMemoryId] = useState(null);
            const [memoryText, setMemoryText] = useState('');
            const [memoryLimits, setMemoryLimits] = useState({ max_items: 20, max_text_length: 500 });
            const [memoryLoading, setMemoryLoading] = useState(false);
            const [memorySaving, setMemorySaving] = useState(false);
            const [memoryMsg, setMemoryMsg] = useState('');
            const [prefViewMode, setPrefViewMode] = useState(localStorage.getItem('viewMode') || 'list');
            const [prefWebSearch, setPrefWebSearch] = useState(localStorage.getItem('webSearch') !== 'false');
            const [prefGroupBy, setPrefGroupBy] = useState(localStorage.getItem('prefGroupBy') === 'mime' ? 'mime' : 'none');
            const [modelsConfig, setModelsConfig] = useState(null);
            const [modelsDraft, setModelsDraft] = useState(null);
            const [modelsConfigError, setModelsConfigError] = useState('');
            const [modelsSaving, setModelsSaving] = useState(false);
            const [scopeCustomMode, setScopeCustomMode] = useState(false);
            const [topkCustomMode, setTopkCustomMode] = useState(false);
            const [modelsSaveMsg, setModelsSaveMsg] = useState('');
            const [prefsConfig, setPrefsConfig] = useState(null);
            const [prefsDraft, setPrefsDraft] = useState({ session_timeout_minutes: 120, registration_enabled: false });
            const [prefsSaving, setPrefsSaving] = useState(false);
            const [prefsMsg, setPrefsMsg] = useState('');
            const [prefsError, setPrefsError] = useState('');
            const sortedGraphMemoryItems = useMemo(() => [...graphMemoryItems].sort((left, right) => {
                const statusPriority = { pending: 0, active: 1, rejected: 2 };
                const statusOrder = (statusPriority[left.status] ?? 3) - (statusPriority[right.status] ?? 3);
                if (statusOrder !== 0) return statusOrder;
                const updatedOrder = (Date.parse(right.updated_at || '') || 0) - (Date.parse(left.updated_at || '') || 0);
                return updatedOrder || String(left.id).localeCompare(String(right.id));
            }), [graphMemoryItems]);
            const graphMemoryPageCount = Math.max(1, Math.ceil(sortedGraphMemoryItems.length / GRAPH_MEMORY_PAGE_SIZE));
            const visibleGraphMemoryItems = sortedGraphMemoryItems.slice(
                graphMemoryPage * GRAPH_MEMORY_PAGE_SIZE,
                (graphMemoryPage + 1) * GRAPH_MEMORY_PAGE_SIZE,
            );
            React.useEffect(() => { if (tab === 'security') loadSessions(); }, [tab]);
            React.useEffect(() => { if (tab === 'memory') loadMemory(); }, [tab]);
            React.useEffect(() => { if (userProfile?.username) setEditUsername(userProfile.username); }, [userProfile?.username]);
            React.useEffect(() => { localStorage.setItem('prefGroupBy', prefGroupBy); }, [prefGroupBy]);
            React.useEffect(() => {
                setGraphMemoryPage(page => Math.min(page, graphMemoryPageCount - 1));
            }, [graphMemoryPageCount]);
            React.useEffect(() => {
                if (!highlightedGraphMemoryId) return undefined;
                const scrollTimer = window.setTimeout(() => {
                    document.getElementById(`graph-memory-${highlightedGraphMemoryId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 0);
                const clearTimer = window.setTimeout(() => setHighlightedGraphMemoryId(null), 1800);
                return () => {
                    window.clearTimeout(scrollTimer);
                    window.clearTimeout(clearTimer);
                };
            }, [graphMemoryPage, highlightedGraphMemoryId]);
            React.useEffect(() => {
                if (tab !== 'api_keys') return undefined;
                let cancelled = false;
                const loadAiModels = async () => {
                    try {
                        let config;
                        if (userProfile?.is_admin) {
                            const [adminConfig, accountConfig] = await Promise.all([
                                api.adminGetAiModels(),
                                api.getModelConfig('ollama'),
                            ]);
                            config = mergeAdminAndAccountModelConfig(adminConfig, accountConfig);
                        } else {
                            config = await api.getModelConfig('ollama');
                        }
                        if (cancelled) return;
                        setModelsConfig(config);
                        if (userProfile?.is_admin) {
                            setModelsDraft({
                                chat: {
                                    enabled: config.chat?.enabled !== false,
                                    default_model: config.chat?.default_model || config.chat?.model || '',
                                    allowed_models: [...(config.chat?.allowed_models || [])],
                                    fallback_chat_model: config.chat?.fallback_chat_model || null,
                                },
                                vision: { enabled: Boolean(config.vision?.enabled), model: config.vision?.model || null },
                                intelligence: { enabled: Boolean(config.intelligence?.enabled), model: config.intelligence?.model || null },
                                memory_extraction: { enabled: Boolean(config.memory_extraction?.enabled), model: config.memory_extraction?.model || null },
                                reranker: { enabled: Boolean(config.reranker?.enabled) },
                                max_num_ctx: Number(config.model_max_num_ctx) || 16384,
                                file_scope: Number(config.file_scope) || 5,
                                top_k: Number(config.top_k) || 20,
                                search_depth: config.search_depth || 'conservative',
                            });
                        }
                        setModelsConfigError('');
                    } catch (error) {
                        if (!cancelled) setModelsConfigError(error.message || 'Failed to load AI model status');
                    }
                };
                const modelChanged = () => loadAiModels();
                const storageChanged = (event) => {
                    if (event.key === 'lavix_model_revision') loadAiModels();
                };
                loadAiModels();
                window.addEventListener('model-changed', modelChanged);
                window.addEventListener('storage', storageChanged);
                return () => {
                    cancelled = true;
                    window.removeEventListener('model-changed', modelChanged);
                    window.removeEventListener('storage', storageChanged);
                };
            }, [tab, userProfile?.is_admin]);


            const loadSessions = async () => {
                setSessionsLoading(true);
                try { setSessions(await api.getSessions()); } catch(e) {} finally { setSessionsLoading(false); }
            };
            const loadMemory = async () => {
                setMemoryLoading(true);
                setMemoryMsg('');
                try {
                    const [data, graph, graphItems] = await Promise.all([
                        api.getMemory(),
                        api.getGraphMemory(),
                        api.getAllGraphMemoryItems(),
                    ]);
                    const consent = resolveMemoryConsent(data.enabled, graph.enabled);
                    setMemoryConsentMismatch(consent.mismatch);
                    // Fail closed when a pre-migration compatibility flag and
                    // the authoritative graph setting disagree. The canonical
                    // toggle below writes both values atomically and repairs it.
                    setMemoryEnabled(consent.enabled);
                    setMemoryItems(Array.isArray(data.items) ? data.items : []);
                    setGraphMemory(graph);
                    setGraphMemoryItems(Array.isArray(graphItems.items) ? graphItems.items : []);
                    setMemoryLimits({
                        max_items: Number(data.max_items) || 20,
                        max_text_length: Number(data.max_text_length) || 500,
                    });
                    if (consent.mismatch) {
                        setMemoryMsg('Memory consent is out of sync. Use Safe Automatic to repair it before saving preferences.');
                    }
                } catch (e) {
                    setMemoryMsg(e.message || 'Failed to load memory');
                } finally {
                    setMemoryLoading(false);
                }
            };
            const toggleMemory = async () => {
                const next = !memoryEnabled;
                setMemorySaving(true);
                setMemoryMsg('');
                try {
                    const updated = await api.updateGraphMemory({
                        enabled: next,
                        retentionDays: graphMemory?.retention_days || 90,
                        expectedRevision: graphMemory?.revision ?? 0,
                    });
                    setGraphMemory(updated);
                    setMemoryEnabled(Boolean(updated.enabled));
                    setMemoryConsentMismatch(false);
                    setMemoryMsg(next ? 'Safe Automatic memory enabled' : 'Memory disabled; existing memories were kept until expiry');
                } catch (e) {
                    setMemoryMsg(e.message || 'Failed to update memory');
                } finally {
                    setMemorySaving(false);
                }
            };
            const updateMemoryRetention = async (retentionDays) => {
                if (!graphMemory) return;
                setMemorySaving(true);
                setMemoryMsg('');
                try {
                    const updated = await api.updateGraphMemory({
                        enabled: graphMemory.enabled,
                        retentionDays,
                        expectedRevision: graphMemory.revision,
                    });
                    setGraphMemory(updated);
                    setMemoryEnabled(Boolean(updated.enabled));
                    setMemoryConsentMismatch(false);
                    setMemoryMsg(`Memory retention set to ${retentionDays} days`);
                } catch (error) {
                    setMemoryMsg(error.message || 'Failed to update memory retention');
                } finally {
                    setMemorySaving(false);
                }
            };
            const mutateGraphItem = async (action, item) => {
                setMemorySaving(true);
                setMemoryMsg('');
                try {
                    if (action === 'approve') await api.approveGraphMemoryItem(item.id);
                    else if (action === 'renew') await api.renewGraphMemoryItem(item.id);
                    else await api.deleteGraphMemoryItem(item.id);
                    if (editingGraphMemoryId === item.id) setEditingGraphMemoryId(null);
                    await loadMemory();
                } catch (error) {
                    setMemoryMsg(error.message || 'Failed to update relationship memory');
                } finally {
                    setMemorySaving(false);
                }
            };
            const beginGraphMemoryEdit = item => {
                setEditingGraphMemoryId(item.id);
                setGraphMemoryDraft({
                    subject: item.subject || '',
                    predicate: item.predicate || '',
                    objectValue: item.object_value || '',
                });
                setMemoryMsg('');
            };
            const saveGraphMemoryEdit = async item => {
                const subject = graphMemoryDraft.subject.trim();
                const predicate = graphMemoryDraft.predicate.trim();
                const objectValue = graphMemoryDraft.objectValue.trim();
                if (!subject || !predicate || !objectValue) {
                    setMemoryMsg('Subject, relationship, and value are required');
                    return;
                }
                setMemorySaving(true);
                setMemoryMsg('');
                try {
                    await api.editGraphMemoryItem(item.id, {
                        subject,
                        predicate,
                        objectValue,
                        expectedRevision: item.revision,
                    });
                    setEditingGraphMemoryId(null);
                    await loadMemory();
                    setMemoryMsg('Relationship memory updated');
                } catch (error) {
                    setMemoryMsg(error.message || 'Failed to edit relationship memory');
                } finally {
                    setMemorySaving(false);
                }
            };

            const formatMemoryDate = value => {
                if (!value) return '';
                const date = new Date(value);
                return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString();
            };
            const addMemory = async () => {
                const value = memoryText.trim();
                if (!value || !memoryEnabled) return;
                setMemorySaving(true);
                setMemoryMsg('');
                try {
                    const data = await api.addMemory(value);
                    setMemoryItems(items => [data.item, ...items]);
                    setMemoryText('');
                    setMemoryMsg('Preference saved');
                } catch (e) {
                    setMemoryMsg(e.message || 'Failed to add memory');
                } finally {
                    setMemorySaving(false);
                }
            };
            const saveAiModels = async () => {
                if (!userProfile?.is_admin || !modelsDraft || !modelsConfig) return;
                setModelsSaving(true);
                setModelsSaveMsg('');
                setModelsConfigError('');
                try {
                    const adminConfig = await api.adminUpdateAiModels({
                        expected_revision: Number(modelsConfig.revision) || 0,
                        chat: (({ fallback_chat_model, ...rest }) => rest)(modelsDraft.chat),
                        vision: modelsDraft.vision,
                        intelligence: modelsDraft.intelligence,
                        memory_extraction: modelsDraft.memory_extraction,
                        reranker: modelsDraft.reranker,
                        model_max_num_ctx: Number(modelsDraft.max_num_ctx) || null,
                        fallback_chat_model: modelsDraft.chat?.fallback_chat_model || null,
                        file_scope: Number(modelsDraft.file_scope) || null,
                        top_k: Number(modelsDraft.top_k) || null,
                        search_depth: modelsDraft.search_depth || null,
                    });
                    const accountConfig = await api.getModelConfig('ollama');
                    const updated = mergeAdminAndAccountModelConfig(adminConfig, accountConfig);
                    setModelsConfig(updated);
                    setModelsDraft({
                        chat: {
                            enabled: updated.chat?.enabled !== false,
                            default_model: updated.chat?.default_model || '',
                            allowed_models: [...(updated.chat?.allowed_models || [])],
                            fallback_chat_model: updated.chat?.fallback_chat_model || null,
                        },
                        vision: { enabled: Boolean(updated.vision?.enabled), model: updated.vision?.model || null },
                        intelligence: { enabled: Boolean(updated.intelligence?.enabled), model: updated.intelligence?.model || null },
                        memory_extraction: { enabled: Boolean(updated.memory_extraction?.enabled), model: updated.memory_extraction?.model || null },
                        reranker: { enabled: Boolean(updated.reranker?.enabled) },
                        max_num_ctx: Number(updated.model_max_num_ctx) || 16384,
                        file_scope: Number(updated.file_scope) || 5,
                        top_k: Number(updated.top_k) || 20,
                        search_depth: updated.search_depth || 'conservative',
                    });
                    localStorage.setItem('lavix_model_revision', String(updated.revision || Date.now()));
                    window.dispatchEvent(new CustomEvent('model-changed'));
                    setModelsSaveMsg('AI model configuration saved');
                } catch (error) {
                    setModelsConfigError(error.message || 'Failed to update AI model configuration');
                } finally {
                    setModelsSaving(false);
                }
            };
            React.useEffect(() => {
                if (tab !== 'admin_preferences' || !userProfile?.is_admin) return undefined;
                let cancelled = false;
                const loadPreferences = async () => {
                    try {
                        const config = await api.adminGetPreferences();
                        if (cancelled) return;
                        setPrefsConfig(config);
                        setPrefsDraft({
                            session_timeout_minutes: Number(config.session_timeout_minutes) || 120,
                            registration_enabled: config.registration_enabled === true,
                        });
                        setPrefsError('');
                    } catch (error) {
                        if (!cancelled) setPrefsError(error.message || 'Failed to load preferences');
                    }
                };
                loadPreferences();
                return () => { cancelled = true; };
            }, [tab, userProfile?.is_admin]);
            const savePreferences = async () => {
                if (!userProfile?.is_admin || !prefsConfig) return;
                setPrefsSaving(true);
                setPrefsMsg('');
                setPrefsError('');
                try {
                    const updated = await api.adminUpdatePreferences({
                        expected_revision: Number(prefsConfig.expected_revision) || 0,
                        session_timeout_minutes: Number(prefsDraft.session_timeout_minutes),
                        registration_enabled: prefsDraft.registration_enabled === true,
                    });
                    setPrefsConfig(updated);
                    setPrefsDraft({
                        session_timeout_minutes: Number(updated.session_timeout_minutes) || 120,
                        registration_enabled: updated.registration_enabled === true,
                    });
                    setPrefsMsg('Preferences saved — new sessions use them immediately');
                } catch (error) {
                    setPrefsError(error.message || 'Failed to update preferences');
                } finally {
                    setPrefsSaving(false);
                }
            };
            const savePersona = async () => {
                setSaving(true);
                try {
                    await api.updatePersona(persona);
                    setSaveMsg('Saved!');
                    onProfileUpdate && onProfileUpdate({ ...userProfile, persona_prompt: persona });
                } catch(e) { setSaveMsg('Failed to save.'); }
                finally { setSaving(false); setTimeout(() => setSaveMsg(''), 3000); }
            };
            const saveUsername = async () => {
                const trimmed = editUsername.trim();
                if (!trimmed) return;
                setUsernameSaving(true);
                setUsernameMsg('');
                try {
                    await api.updateUsername(trimmed);
                    localStorage.setItem('username', trimmed);
                    onProfileUpdate && onProfileUpdate({ ...userProfile, username: trimmed });
                    onUsernameChange && onUsernameChange(trimmed);
                    setUsernameMsg('Username updated!');
                } catch(e) { setUsernameMsg(e.message || 'Failed to update username'); }
                finally { setUsernameSaving(false); setTimeout(() => setUsernameMsg(''), 4000); }
            };
            const changePassword = async () => {
                if (newPw !== confirmPw) { setPwMsg('Passwords do not match'); return; }
                if (newPw.length < 8) { setPwMsg('Minimum 8 characters'); return; }
                try {
                    await api.changePassword(currentPw, newPw);
                    setPwMsg('Password changed!');
                    setCurrentPw(''); setNewPw(''); setConfirmPw('');
                } catch(e) { setPwMsg(e.message || 'Failed'); }
                setTimeout(() => setPwMsg(''), 4000);
            };
            const settingsTabs = [
                { id: 'profile', label: 'Profile', icon: Icons.User },
                { id: 'security', label: 'Security', icon: Icons.Shield },
                { id: 'prefs', label: 'Preferences', icon: Icons.Sliders },
                { id: 'memory', label: 'AI Memory', icon: Icons.Database },
                { id: 'interface', label: 'Interface', icon: Icons.Monitor },
                ...(userProfile?.is_admin ? [
                    { id: '_divider', divider: true, label: 'ADMIN' },
                    { id: 'api_keys', label: 'AI Models', icon: Icons.Key },
                    { id: 'admin_preferences', label: 'Preferences', icon: Icons.Settings },
                    { id: 'admin_users', label: 'Users', icon: Icons.User },
                    { id: 'admin_storage', label: 'Storage', icon: Icons.HardDrive },
                    { id: 'admin_system', label: 'System', icon: Icons.Settings },
                ] : []),
            ];
            return (
                <div className="h-full flex flex-col overflow-hidden">
                    <div className="flex-shrink-0 px-8 py-6 border-b border-white/5">
                        <h1 className="text-2xl font-bold text-white">Settings</h1>
                        <p className="text-sm text-gray-500 mt-0.5">Manage your account and preferences</p>
                    </div>
                    <div className="flex-1 flex overflow-hidden">
                        <div className="w-52 border-r border-white/5 p-4 flex-shrink-0">
                            {settingsTabs.map(t => {
                                if (t.divider) return <div key={t.id} className="px-4 pt-4 pb-1 text-[9px] font-bold text-zinc-600 uppercase tracking-widest">{t.label}</div>;
                                // Guard against an undefined icon component (React #130).
                                const Icon = t.icon || Icons.File;
                                const isAdmin = t.id.startsWith('admin_');
                                return (
                                    <button key={t.id} data-testid={`settings-tab-${t.id}`} onClick={() => setTab(t.id)}
                                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl mb-1 text-sm transition-all ${tab === t.id ? (isAdmin ? 'bg-red-600/15 text-white border border-red-500/20' : 'bg-indigo-500/20 border border-indigo-400/50 text-indigo-200') : 'text-zinc-500 hover:text-white hover:bg-white/5'}`}>
                                        <Icon size={16} />{t.label}
                                    </button>
                                );
                            })}
                        </div>
                        <div className="flex-1 overflow-y-auto p-8">
                            {tab === 'profile' && (
                                <div className="max-w-xl space-y-5">
                                    {/* Avatar section */}
                                    <div>
                                        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Profile Photo</label>
                                        <div className="flex items-center gap-4">
                                            {userProfile?.avatar_data
                                                ? <img src={userProfile.avatar_data} className="w-16 h-16 rounded-full object-cover border border-white/10 flex-shrink-0" alt="avatar" />
                                                : <div className="w-16 h-16 rounded-full bg-red-600/20 flex items-center justify-center text-red-400 font-bold text-2xl border border-white/10 flex-shrink-0">{userProfile?.username?.charAt(0).toUpperCase()}</div>
                                            }
                                            <div className="space-y-1.5">
                                                <button onClick={() => document.getElementById('avatar-file-input').click()}
                                                    className="block px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-white text-xs rounded-lg transition-all">
                                                    Change Photo
                                                </button>
                                                {userProfile?.avatar_data && (
                                                    <button onClick={async () => {
                                                        try { await api.updateAvatar(''); onProfileUpdate && onProfileUpdate({ ...userProfile, avatar_data: null }); setSaveMsg('Photo removed'); }
                                                        catch(e) { setSaveMsg('Failed to remove'); }
                                                        setTimeout(() => setSaveMsg(''), 3000);
                                                    }} className="block px-3 py-1.5 text-xs text-red-400 hover:text-red-300 transition-colors">Remove</button>
                                                )}
                                            </div>
                                        </div>
                                        <input id="avatar-file-input" type="file" accept="image/*" className="hidden" onChange={async (e) => {
                                            const file = e.target.files?.[0];
                                            if (!file) return;
                                            e.target.value = '';
                                            const reader = new FileReader();
                                            reader.onload = async (ev) => {
                                                const img = new Image();
                                                img.onload = async () => {
                                                    const canvas = document.createElement('canvas');
                                                    canvas.width = 128; canvas.height = 128;
                                                    const ctx = canvas.getContext('2d');
                                                    const scale = Math.max(128 / img.width, 128 / img.height);
                                                    const w = img.width * scale, h = img.height * scale;
                                                    ctx.drawImage(img, (128 - w) / 2, (128 - h) / 2, w, h);
                                                    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
                                                    try {
                                                        await api.updateAvatar(dataUrl);
                                                        onProfileUpdate && onProfileUpdate({ ...userProfile, avatar_data: dataUrl });
                                                        setSaveMsg('Photo updated!');
                                                    } catch(err) { setSaveMsg('Failed: ' + err.message); }
                                                    setTimeout(() => setSaveMsg(''), 3000);
                                                };
                                                img.src = ev.target.result;
                                            };
                                            reader.readAsDataURL(file);
                                        }} />
                                        {saveMsg && <p className={`text-xs mt-2 ${saveMsg.includes('Failed') ? 'text-red-400' : 'text-green-400'}`}>{saveMsg}</p>}
                                    </div>
                                    <div>
                                        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Username</label>
                                        <div className="flex gap-2">
                                            <input
                                                value={editUsername}
                                                onChange={e => setEditUsername(e.target.value)}
                                                onKeyDown={e => e.key === 'Enter' && !usernameSaving && saveUsername()}
                                                className="input-field flex-1 text-sm"
                                                placeholder="your-username"
                                            />
                                            <button onClick={saveUsername} disabled={usernameSaving || !editUsername.trim()}
                                                className="px-4 py-2 bg-indigo-600/15 border border-indigo-500/30 text-indigo-200 hover:bg-indigo-600/25 hover:text-white text-sm rounded-xl transition-all disabled:opacity-50 whitespace-nowrap">
                                                {usernameSaving ? '…' : 'Save'}
                                            </button>
                                        </div>
                                        {usernameMsg && <p className={`text-xs mt-1.5 ${usernameMsg.includes('updated') ? 'text-green-400' : 'text-red-400'}`}>{usernameMsg}</p>}
                                    </div>
                                    <div>
                                        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Email</label>
                                        <div className="input-field text-sm opacity-50 cursor-not-allowed">{userProfile?.email || '—'}</div>
                                    </div>
                                    <div>
                                        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">AI Persona Prompt</label>
                                        <p className="text-xs text-gray-600 mb-2">Customise how the AI responds to you in every chat session.</p>
                                        <textarea value={persona} onChange={e => setPersona(e.target.value)} rows={6}
                                            placeholder="E.g. You are a concise legal research assistant..."
                                            className="input-field resize-none text-sm w-full" />
                                        <div className="flex items-center gap-3 mt-3">
                                            <button onClick={savePersona} disabled={saving}
                                                className="px-4 py-2 bg-indigo-600/15 border border-indigo-500/30 text-indigo-200 hover:bg-indigo-600/25 hover:text-white text-sm rounded-xl transition-all disabled:opacity-50">
                                                {saving ? 'Saving…' : 'Save Persona'}
                                            </button>
                                            {saveMsg && <span className={`text-sm ${saveMsg.includes('Failed') ? 'text-red-400' : 'text-green-400'}`}>{saveMsg}</span>}
                                        </div>
                                    </div>
                                    <div>
                                        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Default Chat Type</label>
                                        <p className="text-xs text-gray-600 mb-3">Choose which style is active when you open a new chat.</p>
                                        <div className="flex gap-3">
                                            {PROMPT_OPTIONS.map(opt => (
                                                <button key={opt.key} onClick={() => {
                                                    localStorage.setItem('defaultChatType', opt.key);
                                                    setActivePrompt(opt.key);
                                                    localStorage.setItem('activePrompt', opt.key);
                                                }}
                                                    className={`flex-1 px-4 py-3 rounded-xl border text-left transition-all ${
                                                        (localStorage.getItem('defaultChatType') || 'A') === opt.key
                                                            ? 'bg-indigo-500/20 border-indigo-400/50 text-indigo-200'
                                                            : 'bg-white/5 border-white/10 text-zinc-400 hover:bg-white/10 hover:text-white'
                                                    }`}>
                                                    <span className="text-base mr-1.5">{opt.icon}</span>
                                                    <span className="text-sm font-semibold">{opt.label}</span>
                                                    <p className="text-[11px] text-gray-500 mt-1 ml-6">{opt.desc}</p>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}
                            {tab === 'security' && (
                                <div className="max-w-xl space-y-8">
                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-4">Change Password</h3>
                                        <div className="space-y-3">
                                            <input type="password" value={currentPw} onChange={e => setCurrentPw(e.target.value)} placeholder="Current password" className="input-field text-sm" />
                                            <input type="password" value={newPw} onChange={e => setNewPw(e.target.value)} placeholder="New password (min 8 chars)" className="input-field text-sm" />
                                            <input type="password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)} placeholder="Confirm new password" className="input-field text-sm" />
                                            <div className="flex items-center gap-3">
                                                <button onClick={changePassword} className="px-4 py-2 bg-indigo-600/15 border border-indigo-500/30 text-indigo-200 hover:bg-indigo-600/25 hover:text-white text-sm rounded-xl transition-all">Change Password</button>
                                                {pwMsg && <span className={`text-sm ${pwMsg.includes('!') ? 'text-green-400' : 'text-red-400'}`}>{pwMsg}</span>}
                                            </div>
                                        </div>
                                    </div>
                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-4">Active Sessions</h3>
                                        {sessionsLoading ? <div className="text-gray-500 text-sm">Loading…</div> : (
                                            <div className="space-y-2">
                                                {sessions.length === 0 && <div className="text-gray-600 text-sm">No active sessions found.</div>}
                                                {sessions.map(s => (
                                                    <div key={s.session_id} className="flex items-center justify-between p-3 bg-white/[0.03] border border-white/5 rounded-xl">
                                                        <div>
                                                            <div className="text-xs font-mono text-gray-400">{String(s.session_id).slice(0, 8)}…</div>
                                                            <div className="text-[10px] text-gray-600 mt-0.5">Created: {new Date(s.created_at).toLocaleString()}</div>
                                                        </div>
                                                        <button onClick={async () => { await api.revokeSession(s.session_id).catch(() => {}); loadSessions(); }}
                                                            className="px-3 py-1.5 text-xs text-red-400 border border-red-500/20 rounded-lg hover:bg-red-500/10 transition-all">Revoke</button>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                            {tab === 'prefs' && (
                                <div className="max-w-xl space-y-4">
                                    <div className="flex items-center justify-between p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div>
                                            <div className="text-sm font-medium text-white">Default View</div>
                                            <div className="text-xs text-gray-500 mt-0.5">Grid or list view on the dashboard</div>
                                        </div>
                                        <VaultDropdown
                                            value={prefViewMode}
                                            ariaLabel="Default view"
                                            options={[{ value: 'list', label: 'List' }, { value: 'grid', label: 'Grid' }]}
                                            onChange={(v) => { setPrefViewMode(v); localStorage.setItem('viewMode', v); }}
                                        />
                                    </div>
                                    <div className="flex items-center justify-between p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div>
                                            <div className="text-sm font-medium text-white">Web Search Default</div>
                                            <div className="text-xs text-gray-500 mt-0.5">Enable web search by default in AI Chat</div>
                                        </div>
                                        <button onClick={() => { const v = !prefWebSearch; setPrefWebSearch(v); localStorage.setItem('webSearch', String(v)); window.dispatchEvent(new CustomEvent('websearch-changed')); }}
                                            className={`relative w-10 h-6 rounded-full transition-colors ${prefWebSearch ? 'bg-indigo-600' : 'bg-white/10'}`}>
                                            <span className={`absolute top-[3px] left-[3px] w-[18px] h-[18px] bg-white rounded-full shadow transition-transform duration-200 ${prefWebSearch ? 'translate-x-4' : 'translate-x-0'}`} />
                                        </button>
                                    </div>
                                    {/* Group files by */}
                                    <div className="flex items-center justify-between py-4 border-b border-white/5">
                                        <div>
                                            <div className="text-sm font-medium text-white">Group files by</div>
                                            <div className="text-xs text-gray-500 mt-0.5">How files are organized in grid view</div>
                                        </div>
                                        <VaultDropdown
                                            value={prefGroupBy}
                                            ariaLabel="Group files by"
                                            options={[{ value: 'none', label: 'No grouping (default)' }, { value: 'mime', label: 'By File Type' }]}
                                            onChange={(v) => { setPrefGroupBy(v); localStorage.setItem('prefGroupBy', v); window.dispatchEvent(new StorageEvent('storage', { key: 'prefGroupBy', newValue: v })); }}
                                        />
                                    </div>
                                    <div className="flex items-center justify-between p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div>
                                            <div className="text-sm font-medium text-white">Regenerate all thumbnails</div>
                                            <div className="text-xs text-gray-500 mt-0.5">Clear all cached PDF page previews and regenerate on next view</div>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => onConfirmAction?.({
                                                title: 'Regenerate all thumbnails?',
                                                message: 'This will clear all cached PDF page previews. Thumbnails will be regenerated when you view files next.',
                                                confirmText: 'Regenerate',
                                                variant: 'info',
                                                onConfirm: async () => {
                                                    window.clearMvPreviewCaches?.();
                                                },
                                            })}
                                            className="min-h-10 flex-none rounded-xl border border-white/10 px-3 text-xs font-semibold text-zinc-300 hover:bg-white/10"
                                        >
                                            Regenerate
                                        </button>
                                    </div>
                                </div>
                            )}

                            {tab === 'memory' && (
                                <div data-testid="graph-memory-settings" className="max-w-2xl space-y-4">
                                    <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div className="flex items-center justify-between gap-4">
                                            <div>
                                                <div className="text-sm font-medium text-white">Safe Automatic memory</div>
                                                <div className="text-xs text-gray-500 mt-0.5">
                                                    Learns explicit relationships and preferences from your messages. Sensitive or uncertain statements are rejected. Eligible, non-sensitive lower-confidence candidates wait 14 days for review.
                                                </div>
                                            </div>
                                            <button
                                                type="button"
                                                data-testid="graph-memory-toggle"
                                                onClick={toggleMemory}
                                                disabled={memoryLoading || memorySaving || !graphMemory}
                                                aria-pressed={memoryEnabled}
                                                className={`relative inline-flex w-10 h-6 flex-shrink-0 rounded-full transition-colors ${memoryEnabled ? 'bg-indigo-600' : 'bg-zinc-700'}`}
                                            >
                                                <span className={`absolute top-[3px] left-[3px] w-[18px] h-[18px] bg-white rounded-full shadow transition-transform duration-200 ${memoryEnabled ? 'translate-x-4' : 'translate-x-0'}`} />
                                            </button>
                                        </div>
                                        {memoryConsentMismatch && (
                                            <div data-testid="graph-memory-consent-mismatch" className="mt-3 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                                                Memory consent is out of sync. Saved Preferences remain disabled until you use the Safe Automatic switch to repair consent.
                                            </div>
                                        )}
                                        {graphMemory && (
                                            <div className="mt-4 flex items-center justify-between gap-3 border-t border-white/5 pt-3">
                                                <div>
                                                    <div className="text-xs font-medium text-zinc-300">Automatic expiry</div>
                                                    <div className="mt-0.5 text-[10px] text-zinc-600">Active memories are renewed only when confirmed again.</div>
                                                </div>
                                                <VaultDropdown
                                                    testid="graph-memory-retention"
                                                    value={String(graphMemory.retention_days)}
                                                    disabled={memorySaving || memoryConsentMismatch}
                                                    ariaLabel="Automatic expiry"
                                                    options={[{ value: '30', label: '30 days' }, { value: '90', label: '90 days' }, { value: '365', label: '365 days' }]}
                                                    onChange={(v) => updateMemoryRetention(Number(v))}
                                                />
                                            </div>
                                        )}
                                    </div>

                                    {graphMemory && (
                                        <div data-testid="graph-memory-about-me" className="p-4 bg-white/[0.03] border border-white/5 rounded-xl space-y-3">
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <div className="text-sm font-medium text-white">About Me</div>
                                                    <div className="mt-0.5 text-xs text-zinc-500">A grounded summary built only from active memories.</div>
                                                </div>
                                                <div className="flex gap-1.5 text-[9px] font-bold uppercase tracking-wide">
                                                    <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-emerald-300">{graphMemory.counts?.active || 0} active</span>
                                                    <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2 py-1 text-amber-300">{graphMemory.counts?.pending || 0} review</span>
                                                </div>
                                            </div>
                                            <div className="group relative rounded-lg border border-white/5 bg-black/20 p-3 text-sm leading-relaxed text-zinc-300">
                                                {graphMemory.about_me?.summary || 'No relationship memories yet.'}
                                                {graphMemory.about_me?.summary && (
                                                    <button
                                                        type="button"
                                                        onClick={async () => {
                                                            try {
                                                                if (navigator.clipboard?.writeText) {
                                                                    await navigator.clipboard.writeText(graphMemory.about_me.summary);
                                                                } else {
                                                                    const ta = document.createElement('textarea');
                                                                    ta.value = graphMemory.about_me.summary;
                                                                    ta.style.position = 'fixed';
                                                                    ta.style.opacity = '0';
                                                                    document.body.appendChild(ta);
                                                                    ta.select();
                                                                    document.execCommand('copy');
                                                                    document.body.removeChild(ta);
                                                                }
                                                                setSummaryCopied(true);
                                                                if (summaryCopiedTimer.current) clearTimeout(summaryCopiedTimer.current);
                                                                summaryCopiedTimer.current = setTimeout(() => setSummaryCopied(false), 2000);
                                                            } catch(e) { /* ignore */ }
                                                        }}
                                                        className={`absolute top-2 right-2 rounded-md border p-1.5 opacity-0 transition-opacity group-hover:opacity-100 ${summaryCopied ? 'border-emerald-400/40 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-black/60 text-zinc-500 hover:border-indigo-400/30 hover:text-indigo-300'}`}
                                                        title={summaryCopied ? 'Copied!' : 'Copy summary'}
                                                    >
                                                        {summaryCopied ? <Icons.Check size={12}/> : <Icons.Copy size={12}/>}
                                                    </button>
                                                )}
                                            </div>
                                            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-zinc-600">
                                                <span>Retention: {graphMemory.retention_days} days</span>
                                                {graphMemory.next_expiry_at && <span>Next expiry: {formatMemoryDate(graphMemory.next_expiry_at)}</span>}
                                                {graphMemory.last_learned_at && <span>Last learned: {formatMemoryDate(graphMemory.last_learned_at)}</span>}
                                            </div>
                                        </div>
                                    )}

                                    {graphMemory && (
                                        <div data-testid="graph-memory-relationships" className="p-4 bg-white/[0.03] border border-white/5 rounded-xl space-y-3">
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <div className="text-sm font-medium text-white">Relationship memory</div>
                                                    <div className="mt-0.5 text-xs text-zinc-500">Review, edit, renew, or remove what Lavix remembers about you.</div>
                                                </div>
                                                <span className="flex-none text-[10px] text-zinc-600">{graphMemoryItems.length} total</span>
                                            </div>
                                            {!memoryLoading && graphMemoryItems.length === 0 && (
                                                <div className="rounded-lg border border-dashed border-white/10 p-4 text-center text-xs text-zinc-600">No relationship memories yet.</div>
                                            )}
                                            {visibleGraphMemoryItems.map(item => {
                                                const confidence = Math.round(Math.max(0, Math.min(1, Number(item.confidence) || 0)) * 100);
                                                const isEditing = editingGraphMemoryId === item.id;
                                                const projectionTone = item.projection_state === 'projected'
                                                    ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
                                                    : item.projection_state === 'failed'
                                                        ? 'border-red-500/20 bg-red-500/10 text-red-300'
                                                        : 'border-amber-500/20 bg-amber-500/10 text-amber-300';
                                                return (
                                                    <div
                                                        id={`graph-memory-${item.id}`}
                                                        key={item.id}
                                                        className={`rounded-lg border bg-black/20 p-3 transition-all ${highlightedGraphMemoryId === item.id ? 'border-indigo-400/60 ring-2 ring-indigo-500/20' : 'border-white/5'}`}
                                                    >
                                                        <div className="flex items-start justify-between gap-3">
                                                            <div className="min-w-0">
                                                                {!isEditing && (
                                                                    <div className="text-sm text-zinc-300">{(() => {
                                                                        const subj = String(item.subject || '').trim();
                                                                        const pred = String(item.predicate || '').replaceAll('_', ' ').trim();
                                                                        const obj = item.object_value || '';
                                                                        if (subj.toLowerCase() === 'i') {
                                                                            const conjugate = { has: 'have', is: 'are', was: 'were', does: 'do', goes: 'go', works: 'work', plays: 'play', watches: 'watch', reads: 'read', writes: 'write', builds: 'build', uses: 'use', likes: 'like', loves: 'love', enjoys: 'enjoy', prefers: 'prefer', manages: 'manage', cooks: 'cook' };
                                                                            const words = pred.split(' ');
                                                                            const verb = conjugate[words[0].toLowerCase()] || words[0];
                                                                            const rest = [verb, ...words.slice(1), obj].join(' ');
                                                                            return <><span className="font-semibold text-white">You</span> {rest}</>;
                                                                        }
                                                                        return <><span className="font-semibold text-white">{subj}</span> {pred} {obj}</>;
                                                                    })()}</div>
                                                                )}
                                                                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[9px] font-semibold uppercase tracking-wide">
                                                                    <span className="rounded-full border border-white/10 px-2 py-0.5 text-zinc-400">{item.kind}</span>
                                                                    <span className={`rounded-full border px-2 py-0.5 ${item.status === 'active' ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/20 bg-amber-500/10 text-amber-300'}`}>{item.status}</span>
                                                                    <span className="rounded-full border border-sky-500/20 bg-sky-500/10 px-2 py-0.5 text-sky-300">Confidence {confidence}%</span>
                                                                    <span className={`rounded-full border px-2 py-0.5 ${projectionTone}`}>Graph {String(item.projection_state || 'pending').replaceAll('_', ' ')}</span>
                                                                </div>
                                                            </div>
                                                            <div className="flex flex-none items-center gap-1">
                                                                {!isEditing && <button type="button" onClick={() => beginGraphMemoryEdit(item)} disabled={memorySaving} aria-label="Edit relationship memory" className="p-1.5 text-zinc-600 hover:text-indigo-300"><Icons.Pencil size={13} /></button>}
                                                                {item.status === 'pending' && <button type="button" onClick={() => mutateGraphItem('approve', item)} disabled={memorySaving} className="rounded-lg border border-emerald-500/20 px-2 py-1 text-[10px] font-semibold text-emerald-300 hover:bg-emerald-500/10">Approve</button>}
                                                                {item.status === 'active' && <button type="button" onClick={() => mutateGraphItem('renew', item)} disabled={memorySaving} className="rounded-lg border border-indigo-500/20 px-2 py-1 text-[10px] font-semibold text-indigo-300 hover:bg-indigo-500/10">Renew</button>}
                                                                <button type="button" onClick={() => mutateGraphItem('delete', item)} disabled={memorySaving} aria-label="Delete relationship memory" className="p-1.5 text-zinc-600 hover:text-red-400"><Icons.X size={13} /></button>
                                                            </div>
                                                        </div>
                                                        {isEditing && (
                                                            <div className="mt-3 space-y-2 rounded-lg border border-indigo-500/15 bg-indigo-500/[0.04] p-3">
                                                                <label className="block">
                                                                    <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-500">Subject</span>
                                                                    <input value={graphMemoryDraft.subject} maxLength={200} onChange={event => setGraphMemoryDraft(draft => ({ ...draft, subject: event.target.value }))} className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-2.5 py-1.5 text-xs text-zinc-200 outline-none focus:border-indigo-500/40" />
                                                                </label>
                                                                <label className="block">
                                                                    <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-500">Relationship</span>
                                                                    <input value={graphMemoryDraft.predicate} maxLength={100} onChange={event => setGraphMemoryDraft(draft => ({ ...draft, predicate: event.target.value }))} className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-2.5 py-1.5 text-xs text-zinc-200 outline-none focus:border-indigo-500/40" />
                                                                </label>
                                                                <label className="block">
                                                                    <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-500">Value</span>
                                                                    <textarea value={graphMemoryDraft.objectValue} maxLength={500} rows={2} onChange={event => setGraphMemoryDraft(draft => ({ ...draft, objectValue: event.target.value }))} className="mt-1 w-full resize-y rounded-lg border border-white/10 bg-black/30 px-2.5 py-1.5 text-xs text-zinc-200 outline-none focus:border-indigo-500/40" />
                                                                </label>
                                                                <div className="flex justify-end gap-2">
                                                                    <button type="button" onClick={() => setEditingGraphMemoryId(null)} disabled={memorySaving} className="rounded-lg border border-white/10 px-2.5 py-1 text-[10px] font-semibold text-zinc-400 hover:bg-white/5">Cancel</button>
                                                                    <button type="button" onClick={() => saveGraphMemoryEdit(item)} disabled={memorySaving || !graphMemoryDraft.subject.trim() || !graphMemoryDraft.predicate.trim() || !graphMemoryDraft.objectValue.trim()} className="rounded-lg border border-indigo-500/25 bg-indigo-500/15 px-2.5 py-1 text-[10px] font-semibold text-indigo-200 hover:bg-indigo-500/25 disabled:opacity-40">Save revision</button>
                                                                </div>
                                                            </div>
                                                        )}
                                                        {item.source_excerpt && (
                                                            <div className="mt-3 border-l-2 border-white/10 pl-3">
                                                                <div className="text-[9px] font-semibold uppercase tracking-wider text-zinc-600">Source excerpt</div>
                                                                <blockquote className="mt-1 text-xs italic leading-relaxed text-zinc-500">“{item.source_excerpt}”</blockquote>
                                                            </div>
                                                        )}
                                                        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-zinc-600">
                                                            <span>Memory {String(item.id).slice(0, 8)}</span>
                                                            {item.source_chat_id && <span>Chat {String(item.source_chat_id).slice(0, 8)}</span>}
                                                            {item.source_message_id && <span>Message {String(item.source_message_id).slice(0, 8)}</span>}
                                                            {item.source_created_at && <span>Source: {formatMemoryDate(item.source_created_at)}</span>}
                                                            <span>Expires: {formatMemoryDate(item.expires_at)}</span>
                                                            <span>Revision {item.revision}</span>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                            {graphMemoryItems.length > 0 && (
                                                <div className="flex items-center justify-between border-t border-white/5 pt-3">
                                                    <button type="button" onClick={() => setGraphMemoryPage(page => Math.max(0, page - 1))} disabled={graphMemoryPage === 0} aria-label="Previous relationship memory page" className="flex items-center gap-1 rounded-lg border border-white/10 px-2.5 py-1 text-[10px] font-semibold text-zinc-400 hover:bg-white/5 disabled:opacity-30"><Icons.ChevronLeft size={13} /> Previous</button>
                                                    <span className="text-[10px] text-zinc-600">Page {graphMemoryPage + 1} of {graphMemoryPageCount}</span>
                                                    <button type="button" onClick={() => setGraphMemoryPage(page => Math.min(graphMemoryPageCount - 1, page + 1))} disabled={graphMemoryPage >= graphMemoryPageCount - 1} aria-label="Next relationship memory page" className="flex items-center gap-1 rounded-lg border border-white/10 px-2.5 py-1 text-[10px] font-semibold text-zinc-400 hover:bg-white/5 disabled:opacity-30">Next <Icons.ChevronRight size={13} /></button>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    <div data-testid="graph-memory-pin-preference" className="p-4 bg-white/[0.03] border border-white/5 rounded-xl space-y-3">
                                        <div>
                                            <div className="text-sm font-medium text-white">Pin a preference</div>
                                            <div className="text-xs text-gray-500 mt-0.5">
                                                Explicit preferences stay under your control and are included with relationship memory. Never store passwords or secrets.
                                            </div>
                                        </div>
                                        <textarea
                                            data-testid="graph-memory-preference-input"
                                            value={memoryText}
                                            onChange={e => setMemoryText(e.target.value.slice(0, memoryLimits.max_text_length))}
                                            disabled={!memoryEnabled || memorySaving}
                                            rows={3}
                                            maxLength={memoryLimits.max_text_length}
                                                placeholder={memoryEnabled ? 'I prefer concise answers with metric units.' : 'Enable Safe Automatic memory to add preferences.'}
                                            className="input-field resize-none text-sm w-full disabled:opacity-50"
                                        />
                                        <div className="flex items-center justify-between gap-3">
                                            <span className="text-[10px] text-zinc-600">{memoryText.length}/{memoryLimits.max_text_length}</span>
                                            <button
                                                type="button"
                                                data-testid="graph-memory-preference-save"
                                                onClick={addMemory}
                                                disabled={!memoryEnabled || memorySaving || !memoryText.trim() || memoryItems.length >= memoryLimits.max_items}
                                                className="px-4 py-2 bg-indigo-600/15 border border-indigo-500/30 text-indigo-200 hover:bg-indigo-600/25 hover:text-white text-sm rounded-xl transition-all disabled:opacity-50"
                                            >
                                                Save preference
                                            </button>
                                        </div>
                                    </div>

                                    <div data-testid="graph-memory-saved-preferences" className="p-4 bg-white/[0.03] border border-white/5 rounded-xl space-y-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="text-sm font-medium text-white">
                                                Saved preferences ({memoryItems.length}/{memoryLimits.max_items})
                                            </div>
                                        </div>
                                        {memoryLoading && <div className="text-xs text-gray-500">Loading…</div>}
                                        {!memoryLoading && memoryItems.length === 0 && (
                                            <div className="text-xs text-gray-600">No preferences saved.</div>
                                        )}
                                        {memoryItems.map(item => (
                                            <div key={item.id} data-testid={`graph-memory-preference-${item.id}`} className="flex items-start justify-between gap-3 p-3 bg-black/20 border border-white/5 rounded-lg">
                                                <div className="text-sm text-zinc-300 break-words min-w-0">{item.text}</div>
                                                <button
                                                    type="button"
                                                    aria-label="Delete saved preference"
                                                    onClick={async () => {
                                                        setMemorySaving(true);
                                                        try {
                                                            await api.deleteMemoryItem(item.id);
                                                            setMemoryItems(items => items.filter(value => value.id !== item.id));
                                                        } catch (e) {
                                                            setMemoryMsg(e.message || 'Failed to delete memory');
                                                        } finally {
                                                            setMemorySaving(false);
                                                        }
                                                    }}
                                                    disabled={memorySaving}
                                                    className="p-1 text-zinc-600 hover:text-red-400 disabled:opacity-50"
                                                >
                                                    <Icons.X size={14} />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                    <div className="flex flex-col items-stretch justify-between gap-4 rounded-xl border border-amber-500/15 bg-amber-500/[0.04] p-4 sm:flex-row sm:items-center">
                                        <div>
                                            <div className="text-sm font-medium text-white">Clear relationship memory</div>
                                            <div className="mt-0.5 text-xs text-zinc-500">Deletes learned relationships and the About Me summary. Saved preferences are kept.</div>
                                        </div>
                                        <button
                                            type="button"
                                            data-testid="graph-memory-clear-relationships"
                                            onClick={async () => {
                                                if (!window.confirm('Clear relationship memory? Saved preferences will be kept.')) return;
                                                setMemorySaving(true);
                                                setMemoryMsg('');
                                                try {
                                                    await api.clearGraphMemory();
                                                    await loadMemory();
                                                    setMemoryMsg('Relationship memory cleared; saved preferences kept');
                                                } catch (error) {
                                                    setMemoryMsg(error.message || 'Failed to clear relationship memory');
                                                } finally {
                                                    setMemorySaving(false);
                                                }
                                            }}
                                            disabled={memorySaving || memoryLoading}
                                            className="min-h-10 flex-none rounded-xl border border-amber-500/25 px-3 text-xs font-semibold text-amber-300 hover:bg-amber-500/10 disabled:opacity-40"
                                        >
                                            Clear Relationship Memory
                                        </button>
                                    </div>
                                    <div className="flex flex-col items-stretch justify-between gap-4 rounded-xl border border-red-500/15 bg-red-500/[0.04] p-4 sm:flex-row sm:items-center">
                                        <div>
                                            <div className="text-sm font-medium text-white">Clear all personal memory</div>
                                            <div className="mt-0.5 text-xs text-zinc-500">Deletes relationship memory and saved preferences. Note: Documents, indexes, and chat history are not changed.</div>
                                        </div>
                                        <button
                                            type="button"
                                            data-testid="graph-memory-clear"
                                            data-clear-scope="all-personal"
                                            onClick={async () => {
                                                if (!window.confirm('Clear all personal memory? This deletes relationship memory and saved preferences. Note: Documents, indexes, and chat history are not changed.')) return;
                                                setMemorySaving(true);
                                                setMemoryMsg('');
                                                try {
                                                    await api.clearPersonalMemory();
                                                    await loadMemory();
                                                    setMemoryMsg('All personal memory cleared');
                                                } catch (error) {
                                                    setMemoryMsg(error.message || 'Failed to clear personal memory');
                                                } finally {
                                                    setMemorySaving(false);
                                                }
                                            }}
                                            disabled={memorySaving || memoryLoading}
                                            className="min-h-10 flex-none rounded-xl border border-red-500/25 px-3 text-xs font-semibold text-red-300 hover:bg-red-500/10 disabled:opacity-40"
                                        >
                                            Clear all personal memory
                                        </button>
                                    </div>
                                    {memoryMsg && (
                                        <p data-testid="graph-memory-message" className={`text-xs ${/failed|error|out of sync/i.test(memoryMsg) ? 'text-red-400' : 'text-green-400'}`}>{memoryMsg}</p>
                                    )}
                                </div>
                            )}

                            {/* ── INTERFACE TAB ── */}
                            {tab === 'interface' && (
                                <div className="max-w-xl space-y-3">
                                    <p className="text-xs text-gray-500 mb-4">Customize the look, feel, and behaviour of the chat interface.</p>
                                    {[{label:'Font Size', desc:'Chat message text size', key:'fontSize', opts:[{v:'sm',l:'Small'},{v:'md',l:'Medium'},{v:'lg',l:'Large'}], val:prefFontSize, set:v=>{setPrefFontSize(v);localStorage.setItem('fontSize',v);document.documentElement.style.setProperty('--chat-font-size', v==='sm'?'13px':v==='lg'?'17px':'15px');}}].map(row=>(
                                        <div key={row.key} className="flex items-center justify-between p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                            <div><div className="text-sm font-medium text-white">{row.label}</div><div className="text-xs text-gray-500 mt-0.5">{row.desc}</div></div>
                                            <div className="flex gap-1">{row.opts.map(o=>(
                                                <button key={o.v} onClick={()=>row.set(o.v)}
                                                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${row.val===o.v?'bg-indigo-600/15 border border-indigo-500/30 text-indigo-200':'bg-white/5 text-gray-400 hover:bg-white/10'}`}>{o.l}</button>
                                            ))}</div>
                                        </div>
                                    ))}
                                    {[
                                        {label:'Send on Enter', desc:'Press Enter to send, Shift+Enter for new line', val:prefSendOnEnter, set:v=>{setPrefSendOnEnter(v);localStorage.setItem('sendOnEnter',String(v));}},
                                        {label:'Show Message Timestamps', desc:'Display time under each chat message', val:prefTimestamps, set:v=>{setPrefTimestamps(v);localStorage.setItem('showTimestamps',String(v));}},
                                        {label:'Auto-scroll to Bottom', desc:'Automatically scroll down during streaming', val:prefAutoScroll, set:v=>{setPrefAutoScroll(v);localStorage.setItem('autoScroll',String(v));}},
                                    ].map(row=>(
                                        <div key={row.label} className="flex items-center justify-between p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                            <div><div className="text-sm font-medium text-white">{row.label}</div><div className="text-xs text-gray-500 mt-0.5">{row.desc}</div></div>
                                            <button onClick={()=>row.set(!row.val)} className={`relative w-10 h-6 rounded-full transition-colors ${row.val?'bg-indigo-600':'bg-white/10'}`}>
                                                <span className={`absolute top-[3px] left-[3px] w-[18px] h-[18px] bg-white rounded-full shadow transition-transform duration-200 ${row.val?'translate-x-4':'translate-x-0'}`}/>
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* ── AI MODELS TAB ── */}
                            {tab === 'admin_preferences' && userProfile?.is_admin && (
                                <div data-testid="admin-preferences" className="max-w-3xl space-y-5">
                                    <div>
                                        <p className="text-xs text-gray-500">Site-wide runtime preferences. Saved values apply immediately with no restart; unset values fall back to the server configuration.</p>
                                    </div>
                                    {prefsError && (
                                        <p className="text-xs text-red-400">{prefsError}</p>
                                    )}
                                    <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div className="flex items-center justify-between gap-4">
                                            <div>
                                                <div className="text-sm font-medium text-white">Session timeout</div>
                                                <div className="text-xs text-gray-500 mt-0.5">
                                                    Idle lock and access-token lifetime for newly issued sessions. Already-issued tokens keep their lifetime.
                                                    {!prefsConfig?.session_timeout_saved && ' Currently using the server default.'}
                                                </div>
                                            </div>
                                            <VaultDropdown
                                                testid="admin-prefs-timeout"
                                                value={String(prefsDraft.session_timeout_minutes)}
                                                disabled={prefsSaving}
                                                ariaLabel="Session timeout"
                                                options={[
                                                    { value: '15', label: '15 minutes' },
                                                    { value: '30', label: '30 minutes' },
                                                    { value: '60', label: '1 hour' },
                                                    { value: '120', label: '2 hours' },
                                                    { value: '240', label: '4 hours' },
                                                    { value: '480', label: '8 hours' },
                                                    { value: '1440', label: '24 hours' },
                                                ]}
                                                onChange={(v) => setPrefsDraft((draft) => ({ ...draft, session_timeout_minutes: Number(v) || 120 }))}
                                            />
                                        </div>
                                    </div>
                                    <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div className="flex items-center justify-between gap-4">
                                            <div>
                                                <div className="text-sm font-medium text-white">Public registration</div>
                                                <div className="text-xs text-gray-500 mt-0.5">
                                                    When disabled, /api/auth/register returns 403 and the login page hides registration.
                                                    {!prefsConfig?.registration_saved && ' Currently using the server default.'}
                                                </div>
                                            </div>
                                            <button
                                                type="button"
                                                data-testid="admin-prefs-registration"
                                                onClick={() => setPrefsDraft((draft) => ({ ...draft, registration_enabled: !(draft.registration_enabled === true) }))}
                                                disabled={prefsSaving}
                                                aria-pressed={prefsDraft.registration_enabled === true}
                                                className={`relative inline-flex w-10 h-6 flex-shrink-0 rounded-full transition-colors ${prefsDraft.registration_enabled === true ? 'bg-indigo-600' : 'bg-zinc-700'}`}
                                            >
                                                <span className={`absolute top-[3px] left-[3px] w-[18px] h-[18px] bg-white rounded-full shadow transition-transform duration-200 ${prefsDraft.registration_enabled === true ? 'translate-x-4' : 'translate-x-0'}`} />
                                            </button>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <button data-testid="save-admin-preferences" type="button" onClick={savePreferences} disabled={prefsSaving || !prefsConfig} className="rounded-xl bg-indigo-600/15 border border-indigo-500/30 px-4 py-2 text-sm font-semibold text-indigo-200 hover:bg-indigo-600/25 hover:text-white disabled:opacity-40">
                                            {prefsSaving ? 'Saving…' : 'Save preferences'}
                                        </button>
                                        {prefsMsg && (
                                            <span className="text-xs text-green-400">{prefsMsg}</span>
                                        )}
                                    </div>
                                </div>
                            )}
                            {tab === 'api_keys' && (
                                <div data-testid="ai-model-settings" className="max-w-3xl space-y-5">
                                    <div>
                                        <p className="text-xs text-gray-500">Chat and Settings use the same server configuration. {userProfile?.is_admin ? 'Administrators can configure model roles here; users choose their active chat model from Chat.' : 'This is a read-only view; choose your active chat model from Chat.'}</p>
                                    </div>

                                    <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                        <div className="flex items-center justify-between gap-4">
                                            <div>
                                                <div className="text-sm font-semibold text-white">Provider</div>
                                                <div className="text-xs text-zinc-500 mt-1">Local models served via Ollama.</div>
                                            </div>
                                            <VaultDropdown
                                                value="ollama"
                                                disabled
                                                ariaLabel="Active AI provider"
                                                options={[{ value: 'ollama', label: 'Ollama (Local)' }]}
                                                onChange={() => {}}
                                            />
                                        </div>
                                    </div>

                                    {userProfile?.is_admin && (
                                        <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                            <div className="flex items-center justify-between gap-4">
                                                <div>
                                                <div className="text-sm font-semibold text-white">Max context</div>
                                                <div className="text-xs text-zinc-500 mt-1">Set to what your hardware supports — 4/6GB VRAM: 16k, 8GB VRAM: 32k. Recommended: 16k (safe default). Higher values eat VRAM (262k OOM-killed this box). Applies on save.</div>
                                                </div>
                                                <VaultDropdown
                                                    testid="admin-max-num-ctx"
                                                    value={String(modelsDraft?.max_num_ctx || modelsConfig?.model_max_num_ctx || 16384)}
                                                    disabled={modelsSaving || !modelsDraft}
                                                    ariaLabel="Maximum model context window"
                                                    options={(modelsConfig?.model_max_num_ctx_options || [16384, 32768, 65536, 131072, 262144]).map(value => {
                                                        const label = value >= 1024 ? `${Math.round(value / 1024)}k` : `${value}`;
                                                        const recommended = value === 16384 ? ' (Recommended)' : value === 32768 ? ' (Recommended for 8GB VRAM)' : '';
                                                        return { value: String(value), label: `${label}${recommended}` };
                                                    })}
                                                    onChange={(picked) => setModelsDraft(previous => ({ ...previous, max_num_ctx: Number(picked) || 16384 }))}
                                                />
                                            </div>
                                        </div>
                                    )}

                                    {userProfile?.is_admin && (
                                        <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl space-y-4">
                                            <div>
                                                <div className="text-sm font-semibold text-white">RAG Scope</div>
                                                <div className="text-xs text-zinc-500 mt-1">File Scope and Top-K are independent: Top-K controls retrieved chunks, File Scope controls distinct files in the final context.</div>
                                            </div>
                                            <div className="flex items-center justify-between gap-4">
                                                <div>
                                                    <div className="text-sm font-medium text-white">File Scope <span className="text-zinc-500 font-normal">[ {modelsDraft?.file_scope ?? modelsConfig?.file_scope ?? 5} ] files</span></div>
                                                    <div className="text-xs text-zinc-500 mt-1">Controls the maximum number of distinct files that can contribute to the final RAG context. This is separate from Top-K, which controls retrieved chunks.</div>
                                                </div>
                                                <div className="flex items-center gap-2 flex-shrink-0">
                                                <VaultDropdown
                                                    testid="admin-file-scope"
                                                    value={(() => { const current = modelsDraft?.file_scope ?? modelsConfig?.file_scope ?? 5; return (scopeCustomMode || ![1, 3, 5, 10, 20, 50, 100].includes(current)) ? 'custom' : String(current); })()}
                                                    disabled={modelsSaving || !modelsDraft}
                                                    ariaLabel="Maximum distinct files in RAG context"
                                                    options={[{ value: '1', label: '1 file' }, { value: '3', label: '3 files' }, { value: '5', label: '5 files (Recommended)' }, { value: '10', label: '10 files' }, { value: '20', label: '20 files' }, { value: '50', label: '50 files' }, { value: '100', label: '100 files' }, { value: 'custom', label: 'Custom…' }]}
                                                    onChange={(picked) => { if (picked === 'custom') { setScopeCustomMode(true); return; } setScopeCustomMode(false); setModelsDraft(previous => ({ ...previous, file_scope: Number(picked) || 5 })); }}
                                                />
                                                {(() => { const current = modelsDraft?.file_scope ?? modelsConfig?.file_scope ?? 5; return (scopeCustomMode || ![1, 3, 5, 10, 20, 50, 100].includes(current)) ? (
                                                <input
                                                    data-testid="admin-file-scope-custom"
                                                    type="number" min={1} max={100} step={1}
                                                    value={current}
                                                    disabled={modelsSaving || !modelsDraft}
                                                    aria-label="Custom maximum distinct files (1-100)"
                                                    onChange={(event) => {
                                                        const parsed = Math.trunc(Number(event.target.value));
                                                        if (!Number.isFinite(parsed)) return;
                                                        const clamped = Math.min(100, Math.max(1, parsed));
                                                        setScopeCustomMode(![1, 3, 5, 10, 20, 50, 100].includes(clamped));
                                                        setModelsDraft(previous => ({ ...previous, file_scope: clamped }));
                                                    }}
                                                    className="w-24 px-3 py-1.5 bg-white/5 border border-white/10 rounded-xl text-sm text-white text-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80 disabled:opacity-40"
                                                />) : null; })()}
                                                </div>
                                            </div>
                                            <div className="flex items-center justify-between gap-4">
                                                <div>
                                                    <div className="text-sm font-medium text-white">Top-K (Retrieval) <span className="text-zinc-500 font-normal">[ {modelsDraft?.top_k ?? modelsConfig?.top_k ?? 20} ]</span></div>
                                                    <div className="text-xs text-zinc-500 mt-1">Controls how many of the most relevant chunks/results are retrieved from the vector database for each query. Higher values can improve recall by considering more candidates, but may increase retrieval and reranking latency. Lower values reduce processing overhead but may miss relevant information. Top-K controls retrieved chunks, not the number of files.</div>
                                                </div>
                                                <div className="flex items-center gap-2 flex-shrink-0">
                                                <VaultDropdown
                                                    testid="admin-top-k"
                                                    value={(() => { const current = modelsDraft?.top_k ?? modelsConfig?.top_k ?? 20; return (topkCustomMode || ![5, 10, 20, 50, 100, 200, 500].includes(current)) ? 'custom' : String(current); })()}
                                                    disabled={modelsSaving || !modelsDraft}
                                                    ariaLabel="Retrieval candidate chunks per query"
                                                    options={[{ value: '5', label: '5 chunks' }, { value: '10', label: '10 chunks' }, { value: '20', label: '20 chunks (Recommended)' }, { value: '50', label: '50 chunks' }, { value: '100', label: '100 chunks' }, { value: '200', label: '200 chunks' }, { value: '500', label: '500 chunks' }, { value: 'custom', label: 'Custom…' }]}
                                                    onChange={(picked) => { if (picked === 'custom') { setTopkCustomMode(true); return; } setTopkCustomMode(false); setModelsDraft(previous => ({ ...previous, top_k: Number(picked) || 20 })); }}
                                                />
                                                {(() => { const current = modelsDraft?.top_k ?? modelsConfig?.top_k ?? 20; return (topkCustomMode || ![5, 10, 20, 50, 100, 200, 500].includes(current)) ? (
                                                <input
                                                    data-testid="admin-top-k-custom"
                                                    type="number" min={1} max={500} step={1}
                                                    value={current}
                                                    disabled={modelsSaving || !modelsDraft}
                                                    aria-label="Custom retrieval candidate chunks (1-500)"
                                                    onChange={(event) => {
                                                        const parsed = Math.trunc(Number(event.target.value));
                                                        if (!Number.isFinite(parsed)) return;
                                                        const clampedK = Math.min(500, Math.max(1, parsed));
                                                        setTopkCustomMode(![5, 10, 20, 50, 100, 200, 500].includes(clampedK));
                                                        setModelsDraft(previous => ({ ...previous, top_k: clampedK }));
                                                    }}
                                                    className="w-24 px-3 py-1.5 bg-white/5 border border-white/10 rounded-xl text-sm text-white text-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/80 disabled:opacity-40"
                                                />) : null; })()}
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {userProfile?.is_admin && (
                                        <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl space-y-4">
                                            <div>
                                                <div className="text-sm font-semibold text-white">Web Search</div>
                                                <div className="text-xs text-zinc-500 mt-1">Controls how thoroughly the web search leg retrieves evidence: number of search legs, results per leg, the evidence pool size, and excerpt length. Applies to all web search, on every model.</div>
                                            </div>
                                            <div className="flex items-center justify-between gap-4">
                                                <div>
                                                    <div className="text-sm font-medium text-white">Search Depth <span className="text-zinc-500 font-normal">[ {modelsDraft?.search_depth ?? modelsConfig?.search_depth ?? 'conservative'} ]</span></div>
                                                    <div className="text-xs text-zinc-500 mt-1">Conservative keeps today's behavior (3 legs, 5 results per leg). Balanced and Deep retrieve more evidence and longer excerpts at the cost of latency.</div>
                                                </div>
                                                <div className="flex items-center gap-2 flex-shrink-0">
                                                    <VaultDropdown
                                                        testid="admin-search-depth"
                                                        value={modelsDraft?.search_depth ?? modelsConfig?.search_depth ?? 'conservative'}
                                                        disabled={modelsSaving || !modelsDraft}
                                                        ariaLabel="Web search depth"
                                                        options={[
                                                            { value: 'conservative', label: 'Conservative' },
                                                            { value: 'balanced', label: 'Balanced' },
                                                            { value: 'deep', label: 'Deep' },
                                                            { value: 'pro', label: 'Pro' },
                                                        ]}
                                                        onChange={(picked) => setModelsDraft(previous => ({ ...previous, search_depth: picked || 'conservative' }))}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {modelsConfigError && (
                                        <div className="px-4 py-3 rounded-xl border bg-red-500/10 text-red-300 border-red-500/20 text-xs">{modelsConfigError}</div>
                                    )}

                                    {!modelsConfig && !modelsConfigError && (
                                        <div className="p-5 text-sm text-zinc-500 bg-white/[0.02] border border-white/5 rounded-xl">Loading AI model status…</div>
                                    )}

                                    {modelsConfig && (() => {
                                        const rawChat = modelsConfig.chat || {};
                                        const chat = {
                                            ...rawChat,
                                            model: rawChat.model || rawChat.default_model || modelsConfig.active_chat_model || modelsConfig.model,
                                            configured: rawChat.configured !== false && Boolean(rawChat.model || rawChat.default_model || modelsConfig.active_chat_model || modelsConfig.model),
                                            available: rawChat.available !== false,
                                        };
                                        const vision = modelsConfig.vision || {
                                            model: modelsConfig.vision_model || modelsConfig.vlm_model,
                                            enabled: Boolean(modelsConfig.vision_enabled),
                                            configured: Boolean(modelsConfig.vision_model || modelsConfig.vlm_model),
                                            available: Boolean(modelsConfig.vision_available),
                                            optional: true,
                                        };
                                        const intelligence = modelsConfig.intelligence || {
                                            model: modelsConfig.intelligence_model || chat.model,
                                            configured: Boolean(modelsConfig.intelligence_model || chat.model),
                                            available: chat.available !== false,
                                        };
                                        const embedding = modelsConfig.embedding || {
                                            model: modelsConfig.embedding_model,
                                            configured: Boolean(modelsConfig.embedding_model),
                                            available: Boolean(modelsConfig.embedding_model),
                                            dimension: modelsConfig.embedding_dimension,
                                        };
                                        const reranker = modelsConfig.reranker || {
                                            model: modelsConfig.reranker_model,
                                            enabled: Boolean(modelsConfig.enable_reranking),
                                            configured: Boolean(modelsConfig.reranker_configured),
                                            available: Boolean(modelsConfig.reranker_configured),
                                            optional: true,
                                        };
                                        const memoryExtraction = modelsConfig.memory_extraction || {
                                            model: modelsConfig.memory_model,
                                            enabled: Boolean(modelsConfig.memory_model),
                                            configured: Boolean(modelsConfig.memory_model),
                                            available: Boolean(modelsConfig.memory_model),
                                            optional: true,
                                        };
                                        const installedModelNames = (modelsConfig.installed_models || [])
                                            .map(item => typeof item === 'string' ? item : item?.name)
                                            .filter(Boolean);
                                        const roles = [
                                            { key: 'chat', label: 'LLM · Chat', detail: 'RAG and web-grounded answers', data: chat, optional: Boolean(chat?.optional) },
                                            { key: 'vision', label: 'VLM · Vision', detail: 'Optional image understanding', data: vision, optional: vision?.optional !== false },
                                            { key: 'intelligence', label: 'Document intelligence', detail: 'Summaries and semantic tags', data: intelligence, optional: Boolean(intelligence?.optional) },
                                            { key: 'memory_extraction', label: 'Memory extraction', detail: 'Optional relationship-memory learning', data: memoryExtraction, optional: true },
                                            { key: 'embedding', label: 'Embedding', detail: 'Vector indexing and retrieval', data: embedding, optional: Boolean(embedding?.optional) },
                                            { key: 'reranker', label: 'Reranker', detail: 'Optional result reordering', data: reranker, optional: reranker?.optional !== false },
                                        ];
                                        return (
                                            <>
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                    {roles.map(({ key, label, detail, data, optional }) => {
                                                        const draftRole = modelsDraft?.[key];
                                                        const enabled = draftRole ? draftRole.enabled !== false : data?.enabled !== false;
                                                        const configured = data?.configured !== false && Boolean(data?.model || key === 'reranker' && data?.configured);
                                                        const available = data?.available !== false;
                                                        const optionalOff = optional && (!enabled || !configured);
                                                        const healthy = enabled && configured && available;
                                                        const badge = optionalOff ? 'Optional · Off' : healthy ? 'Active' : configured ? 'Unavailable' : 'Not configured';
                                                        const badgeClass = optionalOff
                                                            ? 'bg-zinc-500/10 border-zinc-500/20 text-zinc-400'
                                                            : healthy
                                                                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                                                                : 'bg-red-500/10 border-red-500/20 text-red-300';
                                                        return (
                                                            <div key={key} data-testid={`model-role-${key}`} className="p-4 bg-white/[0.03] border border-white/5 rounded-xl min-w-0">
                                                                <div className="flex items-start justify-between gap-3">
                                                                    <div className="min-w-0">
                                                                        <div className="text-sm font-semibold text-white">{label}</div>
                                                                        <div className="text-[11px] text-zinc-500 mt-0.5">{detail}</div>
                                                                    </div>
                                                                    <div className="flex shrink-0 items-center gap-2">
                                                                        <span className={`text-[9px] px-2 py-1 rounded-full border font-bold uppercase tracking-wider ${badgeClass}`}>{badge}</span>
                                                                        {userProfile?.is_admin && draftRole && key !== 'embedding' && (
                                                                            <button
                                                                                type="button"
                                                                                aria-label={`${enabled ? 'Disable' : 'Enable'} ${label}`}
                                                                                aria-pressed={enabled}
                                                                                onClick={() => setModelsDraft(previous => ({ ...previous, [key]: { ...previous[key], enabled: !enabled } }))}
                                                                                className={`relative h-5 w-9 rounded-full transition-colors ${enabled ? 'bg-indigo-600' : 'bg-white/10'}`}
                                                                            >
                                                                                <span className={`absolute top-1 h-3 w-3 rounded-full bg-white transition-all ${enabled ? 'left-5' : 'left-1'}`} />
                                                                            </button>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                                {key === 'chat' && userProfile?.is_admin && (
                                                                    <div data-testid="admin-account-chat-model" className="mt-3 space-y-1 rounded-lg border border-white/5 bg-black/20 px-3 py-2 text-[10px]">
                                                                        <div className="flex items-center justify-between gap-3">
                                                                            <span className="text-zinc-500">Your active model</span>
                                                                            <span data-testid="admin-active-chat-model" className="min-w-0 truncate font-mono text-zinc-200" title={modelsConfig.active_chat_model || ''}>{modelsConfig.active_chat_model || 'Unavailable'}</span>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {key === 'chat' && userProfile?.is_admin && modelsDraft && (
                                                                    <div className="mt-3">
                                                                    <VaultDropdown
                                                                        testid="model-role-select-chat"
                                                                        value={modelsDraft.chat?.default_model || ''}
                                                                        disabled={!enabled || modelsSaving}
                                                                        ariaLabel="Select default chat model"
                                                                        options={[{ value: '', label: 'Select default…' }, ...installedModelNames]}
                                                                        onChange={(model) => setModelsDraft(previous => {
                                                                            if (!model) return previous;
                                                                            const current = previous.chat.allowed_models || [];
                                                                            return {
                                                                                ...previous,
                                                                                chat: {
                                                                                    ...previous.chat,
                                                                                    default_model: model,
                                                                                    allowed_models: current.includes(model) ? current : [...current, model],
                                                                                },
                                                                            };
                                                                        })}
                                                                    />
                                                                    </div>
                                                                )}
                                                                {key === 'chat' && userProfile?.is_admin && modelsDraft && (
                                                                    <div className="mt-3 space-y-1 rounded-lg border border-white/5 bg-black/20 px-3 py-2 text-[10px]">
                                                                        <div className="flex items-center justify-between gap-3">
                                                                            <span className="text-zinc-500">Fallback model</span>
                                                                            <span data-testid="admin-fallback-chat-model" className="min-w-0 truncate font-mono text-violet-300" title={modelsDraft?.chat?.fallback_chat_model || modelsConfig.chat?.fallback_chat_model || ''}>{modelsDraft?.chat?.fallback_chat_model || modelsConfig.chat?.fallback_chat_model || 'None (fail loudly)'}</span>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {key === 'chat' && userProfile?.is_admin && modelsDraft && (
                                                                    <div className="mt-3">
                                                                    <VaultDropdown
                                                                        testid="model-role-select-fallback"
                                                                        value={modelsDraft.chat?.fallback_chat_model || ''}
                                                                        disabled={!enabled || modelsSaving}
                                                                        ariaLabel="Select fallback chat model"
                                                                        options={[{ value: '', label: 'None (fail loudly)' }, ...(modelsDraft.chat?.allowed_models || modelsConfig.chat?.allowed_models || []).map(model => ({ value: model, label: model }))]}
                                                                        onChange={(model) => setModelsDraft(previous => ({
                                                                            ...previous,
                                                                            chat: { ...previous.chat, fallback_chat_model: model || null },
                                                                        }))}
                                                                    />
                                                                    </div>
                                                                )}
                                                                {userProfile?.is_admin && draftRole && ['vision', 'intelligence', 'memory_extraction'].includes(key) ? (
                                                                    <div className="mt-3">
                                                                    <VaultDropdown
                                                                        testid={`model-role-select-${key}`}
                                                                        value={draftRole.model || ''}
                                                                        disabled={!enabled || modelsSaving}
                                                                        ariaLabel={`${label} model`}
                                                                        options={[{ value: '', label: 'Not configured' }, ...installedModelNames]}
                                                                        onChange={(model) => setModelsDraft(previous => ({
                                                                            ...previous,
                                                                            [key]: {
                                                                                ...previous[key],
                                                                                model: model || null,
                                                                            },
                                                                        }))}
                                                                    />
                                                                    </div>
                                                                ) : (key === 'chat' && userProfile?.is_admin && modelsDraft) ? null : (
                                                                    <div className="mt-3 font-mono text-xs text-zinc-200 truncate" title={data?.model || ''}>{data?.model || (optionalOff ? 'Not enabled' : 'Unknown')}</div>
                                                                )}
                                                                {key === 'embedding' && data?.dimension && (
                                                                    <div className="mt-2 text-[10px] text-zinc-500">{data.dimension.toLocaleString()} dimensions</div>
                                                                )}
                                                                {key === 'embedding' && data?.change_requires_reembed && (
                                                                    <div data-testid="admin-embedding-reembed-warning" className="mt-2 text-[10px] text-amber-300">Model change needs migration — run: python -m app.cli reembed --model &lt;name&gt; --dimensions &lt;n&gt;</div>
                                                                )}
                                                                {key === 'reranker' && data?.mode && (
                                                                    <div data-testid="admin-reranker-mode" className="mt-2 text-[10px] text-zinc-500">Mode: {data.mode} (deployment choice — change RERANKER_MODE in docker-compose.yaml)</div>
                                                                )}
                                                                {(data?.url || data?.endpoint) && (
                                                                    <div className="mt-2 text-[10px] text-zinc-600 truncate" title={data.url || data.endpoint}>{data.url || data.endpoint}</div>
                                                                )}
                                                            </div>
                                                        );
                                                    })}
                                                </div>

                                                <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                                    <div className="flex items-center justify-between gap-3 mb-3">
                                                        <div>
                                                            <div className="text-sm font-semibold text-white">Allowed chat models</div>
                                                            <div className="text-[11px] text-zinc-500 mt-0.5">{userProfile?.is_admin ? 'This allowlist is the source used by the Chat model menu.' : 'Change your active model from the Chat model menu.'}</div>
                                                        </div>
                                                        <span className="text-[10px] text-zinc-500">{(modelsDraft?.chat?.allowed_models || modelsConfig.allowed_chat_models || modelsConfig.available_models || []).length}</span>
                                                    </div>
                                                    <div className="flex flex-wrap gap-1.5">
                                                        {(userProfile?.is_admin ? (installedModelNames.length > 0 ? installedModelNames : (modelsConfig.deployment_allowed_chat_models || [])) : (modelsConfig.allowed_chat_models || modelsConfig.available_models || [])).map(model => {
                                                            const allowed = (modelsDraft?.chat?.allowed_models || modelsConfig.allowed_chat_models || modelsConfig.available_models || []).includes(model);
                                                            const active = model === (modelsDraft?.chat?.default_model || modelsConfig.chat?.default_model);
                                                            const withinCeiling = !userProfile?.is_admin || !modelsConfig.deployment_allowed_chat_models || (installedModelNames.length > 0 ? modelsConfig.deployment_allowed_chat_models.includes(model) : true);
                                                            return (
                                                                <button
                                                                    type="button"
                                                                    key={model}
                                                                     disabled={!userProfile?.is_admin || modelsSaving}
                                                                    title={active ? 'Default model' : (!withinCeiling ? 'Outside deployment ceiling' : 'Click to add to allowlist')}
                                                            onClick={() => {
                                                                if (!userProfile?.is_admin) return;
                                                                setModelsDraft(previous => {
                                                                    const current = previous.chat.allowed_models || [];
                                                                    if (current.includes(model)) {
                                                                        // Membership toggle only: the default model must be changed
                                                                        // via the "Select default chat model" dropdown first, so
                                                                        // the default can never silently leave the allowlist.
                                                                        if (model === previous.chat.default_model) return previous;
                                                                        return { ...previous, chat: { ...previous.chat, allowed_models: current.filter(item => item !== model) } };
                                                                    }
                                                                    // Membership toggle only: adding a model to the allowlist
                                                                    // must NOT reassign the default model.
                                                                    return { ...previous, chat: { ...previous.chat, allowed_models: [...current, model] } };
                                                                });
                                                            }}
                                                                    className={`text-xs px-2.5 py-1 rounded-full border disabled:cursor-default ${allowed ? 'bg-indigo-500/30 text-indigo-100 border-indigo-400/50 font-medium' : 'bg-white/[0.03] text-zinc-500 border-white/10'} ${!withinCeiling ? 'opacity-40' : ''}`}
                                                                >
                                                                    {userProfile?.is_admin ? (allowed ? '✓ ' : '+ ') : ''}{model}{active && !userProfile?.is_admin ? ' · active' : ''}
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                                {userProfile?.is_admin && modelsDraft && (
                                                    <div className="flex items-center justify-between gap-3 rounded-xl border border-indigo-500/20 bg-indigo-500/[0.05] p-4">
                                                        <div>
                                                            <div className="text-sm font-semibold text-white">Apply model configuration</div>
                                                            <div className="mt-0.5 text-[11px] text-zinc-500">Changes are revision-checked and immediately synchronized with Chat and workers.</div>
                                                            {modelsSaveMsg && <div className="mt-1 text-xs text-emerald-400">{modelsSaveMsg}</div>}
                                                        </div>
                                                        <button data-testid="save-ai-models" type="button" onClick={saveAiModels} disabled={modelsSaving || !(modelsDraft.chat.allowed_models || []).length || !modelsDraft.chat.default_model} className="rounded-xl bg-indigo-600/15 border border-indigo-500/30 px-4 py-2 text-sm font-semibold text-indigo-200 hover:bg-indigo-600/25 hover:text-white disabled:opacity-40">
                                                            {modelsSaving ? 'Saving…' : 'Save changes'}
                                                        </button>
                                                    </div>
                                                )}
                                            </>
                                        );
                                    })()}
                                </div>
                            )}
                            {['admin_users','admin_storage','admin_system'].includes(tab) && (
                                <AdminView forcedTab={tab.replace('admin_', '')} hideSidebar={true} currentUserId={userProfile?.id} />
                            )}
                        </div>
                    </div>
                </div>
            );
        }

        // ── ADMIN VIEW ───────────────────────────────────────────────────────────
        // Permission definitions
        const PERMS = [
            { key: 'perm_upload',   label: 'Upload',   icon: '↑' },
            { key: 'perm_download', label: 'Download', icon: '↓' },
            { key: 'perm_delete',   label: 'Delete',   icon: '🗑' },
            { key: 'perm_ai',       label: 'AI',       icon: '🤖' },
            { key: 'perm_share',    label: 'Copy',     icon: '⧉' },
            { key: 'perm_folders',  label: 'Folders',  icon: '📁' },
            { key: 'perm_rename',   label: 'Rename',   icon: '✏' },
        ];

        function AdminView({ embedded = false, currentUserId = null, hideSidebar = false, forcedTab = null }) {
            const [tab, setTab] = useState(forcedTab || 'users');
            React.useEffect(() => { if (forcedTab) setTab(forcedTab); }, [forcedTab]);
            const [users, setUsers] = useState([]);
            const [usersLoading, setUsersLoading] = useState(false);
            const [systemInfo, setSystemInfo] = useState(null);
            const [storageData, setStorageData] = useState([]);
            const [search, setSearch] = useState('');
            const [resetPwUser, setResetPwUser] = useState(null);
            const [resetPwVal, setResetPwVal] = useState('');
            const [msg, setMsg] = useState('');
            const [expandedUserId, setExpandedUserId] = useState(null);
            const [policies, setPolicies] = useState([]);
            const [togglingPerm, setTogglingPerm] = useState({}); // {userId_permKey: true} for loading state
            const [quotaDraft, setQuotaDraft] = useState({}); // {userId: string GB value} while editing
            const [quotaSaving, setQuotaSaving] = useState({}); // {userId: true} for save-in-flight
            const [clearProcessing, setClearProcessing] = React.useState(false);

            React.useEffect(() => {
                if (tab === 'users') { loadUsers(); loadPolicies(); }
                else if (tab === 'system') loadSystem();
                else if (tab === 'storage') loadStorage();
            }, [tab]);

            React.useEffect(() => {
                if (tab !== 'system') return undefined;
                const interval = window.setInterval(loadSystem, 15000);
                return () => window.clearInterval(interval);
            }, [tab]);

            const loadUsers = async () => { setUsersLoading(true); try { setUsers(await api.adminGetUsers()); } catch(e) {} finally { setUsersLoading(false); } };
            const loadSystem = async () => { try { setSystemInfo(await api.adminGetSystem()); } catch(e) {} };
            const loadStorage = async () => { try { setStorageData(await api.adminGetStorage()); } catch(e) {} };
            const loadPolicies = async () => { try { setPolicies(await api.adminGetPolicies()); } catch(e) {} };
            const toggleActive = async (u) => { await api.adminUpdateUser(u.id, { is_active: !u.is_active }).catch(() => {}); loadUsers(); };
            const toggleAdmin = async (u) => {
                setUsers(prev => prev.map(x => x.id === u.id ? { ...x, is_admin: !x.is_admin } : x));
                await api.adminUpdateUser(u.id, { is_admin: !u.is_admin }).catch(() => loadUsers());
            };
            const doResetPw = async () => {
                if (!resetPwVal || resetPwVal.length < 8) { setMsg('Min 8 characters'); return; }
                try { await api.adminResetPassword(resetPwUser.id, resetPwVal); setMsg('Password reset!'); setResetPwUser(null); setResetPwVal(''); }
                catch(e) { setMsg(e.message || 'Failed'); }
                setTimeout(() => setMsg(''), 3000);
            };

            const togglePerm = async (u, permKey) => {
                const tkKey = `${u.id}_${permKey}`;
                setTogglingPerm(p => ({ ...p, [tkKey]: true }));
                const newVal = !u[permKey];
                // Optimistic update
                setUsers(prev => prev.map(x => x.id === u.id ? { ...x, [permKey]: newVal } : x));
                try {
                    await api.adminSetPermissions(u.id, { [permKey]: newVal });
                } catch(e) {
                    // Revert on failure
                    setUsers(prev => prev.map(x => x.id === u.id ? { ...x, [permKey]: !newVal } : x));
                    setMsg('Failed to update permission');
                    setTimeout(() => setMsg(''), 3000);
                }
                setTogglingPerm(p => { const n = {...p}; delete n[tkKey]; return n; });
            };

            const saveQuota = async (u) => {
                const raw = quotaDraft[u.id];
                const gb = Math.floor(Number(raw));
                if (!Number.isFinite(gb) || gb < 0) { setMsg('Storage quota must be a non-negative whole number of GB'); setTimeout(() => setMsg(''), 3000); return; }
                const qk = `${u.id}`;
                setQuotaSaving(p => ({ ...p, [qk]: true }));
                try {
                    await api.adminUpdateUser(u.id, { storage_quota_gb: gb });
                    setMsg(`Storage quota set to ${gb} GB for ${u.username}`);
                    setQuotaDraft(p => { const n = {...p}; delete n[u.id]; return n; });
                    loadUsers();
                    // Refresh the dashboard STORAGE box underneath the Settings
                    // overlay: the quota changed but no view switch fires.
                    window.dispatchEvent(new CustomEvent('storage-changed'));
                } catch(e) {
                    setMsg(e.message || 'Failed to set storage quota');
                } finally {
                    setQuotaSaving(p => { const n = {...p}; delete n[qk]; return n; });
                }
                setTimeout(() => setMsg(''), 3000);
            };

            const applyPolicy = async (u, policyId) => {
                if (!policyId) return;
                try {
                    await api.adminApplyPolicy(u.id, parseInt(policyId));
                    await loadUsers();
                    setMsg('Policy applied!');
                } catch(e) { setMsg('Failed to apply policy'); }
                setTimeout(() => setMsg(''), 3000);
            };

            // User status indicator
            const userStatusDot = (u) => {
                if (!u.is_active) return 'bg-red-500';
                const allOn = PERMS.every(p => u[p.key] !== false);
                if (allOn) return 'bg-green-400';
                const noneOn = !u.perm_upload && !u.perm_download;
                if (noneOn) return 'bg-red-400';
                return 'bg-yellow-400';
            };

            const filteredUsers = users.filter(u => (u.username + u.email).toLowerCase().includes(search.toLowerCase()));
            const adminTabs = [
                { id: 'users', label: 'Users', icon: Icons.User },
                { id: 'storage', label: 'Storage', icon: Icons.HardDrive },
                { id: 'system', label: 'System', icon: Icons.Settings },
            ];
            return (
                <div className="h-full flex flex-col overflow-hidden">
                    {!embedded && (
                        <div className="flex-shrink-0 px-8 py-6 border-b border-white/5 flex items-center gap-3">
                            <span className="text-red-400"><Icons.Shield size={22} /></span>
                            <div>
                                <h1 className="text-2xl font-bold text-white">Admin Panel</h1>
                                <p className="text-sm text-gray-500 mt-0.5">System administration — restricted access</p>
                            </div>
                        </div>
                    )}
                    <div className="flex-1 flex overflow-hidden">
                        {!hideSidebar && (
                            <div className="w-52 border-r border-white/5 p-4 flex-shrink-0">
                                {adminTabs.map(t => { const Icon = t.icon; return (
                                    <button key={t.id} onClick={() => setTab(t.id)}
                                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl mb-1 text-sm transition-all ${tab === t.id ? 'bg-red-600/15 text-white border border-red-500/20' : 'text-zinc-500 hover:text-white hover:bg-white/5'}`}>
                                        <Icon size={16} />{t.label}
                                    </button>
                                ); })}
                            </div>
                        )}
                        <div className="flex-1 overflow-y-auto p-6">
                            {msg && <div className={`mb-4 px-4 py-2 rounded-xl text-sm border ${msg.includes('!') ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>{msg}</div>}
                            {tab === 'users' && (
                                <div>
                                    <div className="flex items-center gap-3 mb-4">
                                        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search users…" className="input-field text-sm py-2 px-4 max-w-xs" />
                                        <span className="text-gray-500 text-sm">{users.length} total</span>
                                    </div>
                                    {usersLoading ? <div className="text-gray-500 text-sm">Loading users…</div> : (
                                        <div className="space-y-2">
                                            {filteredUsers.map(u => (
                                                <div key={u.id} className={`bg-white/[0.03] border rounded-xl overflow-hidden transition-all ${expandedUserId === u.id ? 'border-indigo-500/30' : 'border-white/5'}`}>
                                                    {/* User row */}
                                                    <div className="flex items-center gap-4 p-4">
                                                        <div className="relative flex-shrink-0">
                                                            <div className="w-9 h-9 rounded-full bg-indigo-600/20 flex items-center justify-center text-indigo-300 font-bold text-sm">{u.username?.charAt(0).toUpperCase()}</div>
                                                            <div className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2 border-black ${userStatusDot(u)}`} title={!u.is_active ? 'Inactive' : PERMS.every(p => u[p.key] !== false) ? 'Full access' : 'Restricted'} />
                                                        </div>
                                                        <div className="flex-1 min-w-0">
                                                            <div className="flex items-center gap-2 flex-wrap">
                                                                <span className="text-sm font-medium text-white">{u.username}</span>
                                                                {u.is_admin && <span className="text-[9px] bg-red-500/20 text-red-400 border border-red-500/20 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">Admin</span>}
                                                                {!u.is_active && <span className="text-[9px] bg-gray-500/20 text-gray-400 px-1.5 py-0.5 rounded uppercase">Inactive</span>}
                                                            </div>
                                                            <div className="text-xs text-gray-500 truncate">{u.email}</div>
                                                            <div className="text-[10px] text-gray-600 mt-0.5">{formatSize(u.storage_used_bytes || 0)} / {formatSize(u.storage_quota_bytes || 10737418240)} · {u.file_count || 0} files</div>
                                                        </div>
                                                        <div className="flex items-center gap-2 flex-shrink-0">
                                                            <button onClick={() => setExpandedUserId(expandedUserId === u.id ? null : u.id)}
                                                                className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${expandedUserId === u.id ? 'bg-indigo-600/20 text-indigo-300 border-indigo-500/30' : 'text-gray-400 border-white/10 hover:bg-white/5'}`}>
                                                                Permissions {expandedUserId === u.id ? '▲' : '▼'}
                                                            </button>
                                                            <button onClick={() => { setResetPwUser(u); setResetPwVal(''); }}
                                                                className="px-3 py-1.5 text-xs text-gray-400 border border-white/10 rounded-lg hover:bg-white/5 transition-all">Reset PW</button>
                                                            <button onClick={() => toggleActive(u)}
                                                                className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${u.is_active ? 'text-yellow-400 border-yellow-500/20 hover:bg-yellow-500/10' : 'text-green-400 border-green-500/20 hover:bg-green-500/10'}`}>
                                                                {u.is_active ? 'Deactivate' : 'Activate'}
                                                            </button>
                                                            {u.id !== currentUserId && (
                                                                <button onClick={() => toggleAdmin(u)}
                                                                    className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${u.is_admin ? 'text-red-400 border-red-500/20 hover:bg-red-500/10' : 'text-zinc-400 border-white/10 hover:bg-white/5'}`}>
                                                                    {u.is_admin ? 'Revoke Admin' : 'Make Admin'}
                                                                </button>
                                                            )}
                                                        </div>
                                                    </div>
                                                    {/* Expanded Permissions Panel */}
                                                    {expandedUserId === u.id && (
                                                        <div className="border-t border-white/5 bg-black/30 px-4 py-4">
                                                            {/* Policy template selector */}
                                                            <div className="flex items-center gap-3 mb-4">
                                                                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Apply Policy:</span>
                                                                <VaultDropdown
                                                                    value=""
                                                                    ariaLabel="Apply policy template"
                                                                    options={[{ value: '', label: '— Select template —' }, ...policies.map(p => ({ value: String(p.id), label: p.name }))]}
                                                                    onChange={(v) => { if (v) applyPolicy(u, v); }}
                                                                />
                                                            </div>
                                                            {/* Permission toggles */}
                                                            <div className="flex flex-wrap gap-2">
                                                                {PERMS.map(p => {
                                                                    const enabled = u[p.key] !== false;
                                                                    const loading = togglingPerm[`${u.id}_${p.key}`];
                                                                    return (
                                                                        <button key={p.key} onClick={() => !loading && togglePerm(u, p.key)}
                                                                            disabled={loading}
                                                                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${enabled ? 'bg-indigo-600/20 text-indigo-300 border-indigo-500/30 hover:bg-indigo-600/30' : 'bg-red-500/10 text-red-400 border-red-500/20 hover:bg-red-500/15'} ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
                                                                            <span>{p.icon}</span>
                                                                            <span>{p.label}</span>
                                                                            <span className={`text-[9px] font-bold uppercase ml-0.5 ${enabled ? 'text-indigo-400' : 'text-red-500'}`}>{enabled ? 'ON' : 'OFF'}</span>
                                                                        </button>
                                                                    );
                                                                })}
                                                            </div>
                                                            {/* Storage quota */}
                                                            <div className="mt-4 pt-4 border-t border-white/5">
                                                                <div className="flex items-center justify-between gap-3 mb-2">
                                                                    <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Storage quota</span>
                                                                    <span className="text-[10px] text-gray-500">{formatSize(u.storage_used_bytes || 0)} used</span>
                                                                </div>
                                                                <div className="h-2 bg-white/5 rounded-full overflow-hidden mb-3">
                                                                    <div className="h-full bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full transition-all"
                                                                        style={{ width: `${Math.min(100, ((u.storage_used_bytes || 0) / Math.max(1, u.storage_quota_bytes || 1)) * 100)}%` }} />
                                                                </div>
                                                                <div className="flex items-center gap-2">
                                                                    <input
                                                                        type="number"
                                                                        min={0}
                                                                        step={1}
                                                                        value={quotaDraft[u.id] ?? Math.floor((u.storage_quota_bytes || 0) / 1073741824)}
                                                                        onChange={e => setQuotaDraft(prev => ({ ...prev, [u.id]: e.target.value }))}
                                                                        aria-label={`Storage quota in GB for ${u.username}`}
                                                                        className="input-field w-28 text-xs py-1.5 px-3"
                                                                    />
                                                                    <span className="text-xs text-gray-400">GB</span>
                                                                    <button
                                                                        onClick={() => saveQuota(u)}
                                                                        disabled={quotaSaving[u.id] || quotaDraft[u.id] === undefined}
                                                                        className="px-3 py-1.5 text-xs rounded-lg border border-indigo-500/30 bg-indigo-600/15 text-indigo-200 hover:bg-indigo-600/25 transition-all disabled:opacity-40"
                                                                    >
                                                                        {quotaSaving[u.id] ? 'Saving…' : 'Set quota'}
                                                                    </button>
                                                                </div>
                                                                <div className="text-[10px] text-gray-600 mt-1.5">0 blocks all uploads. Changes take effect immediately on save.</div>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {resetPwUser && (
                                        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center" onClick={() => setResetPwUser(null)}>
                                            <div className="bg-zinc-900 border border-white/10 rounded-2xl p-6 w-full max-w-sm" onClick={e => e.stopPropagation()}>
                                                <h3 className="text-white font-semibold mb-4">Reset Password — <span className="text-indigo-400">{resetPwUser.username}</span></h3>
                                                <input type="password" value={resetPwVal} onChange={e => setResetPwVal(e.target.value)} placeholder="New password (min 8 chars)" className="input-field text-sm mb-4" />
                                                <div className="flex gap-3">
                                                    <button onClick={doResetPw} className="flex-1 py-2 bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 text-indigo-200 hover:text-white text-sm rounded-xl transition-all">Reset</button>
                                                    <button onClick={() => setResetPwUser(null)} className="flex-1 py-2 bg-white/5 hover:bg-white/10 text-gray-300 text-sm rounded-xl transition-all">Cancel</button>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                            {tab === 'storage' && (
                                <div className="space-y-3">
                                    <h3 className="text-base font-semibold text-white mb-4">Storage by User</h3>
                                    {storageData.map(u => (
                                        <div key={u.user_id} className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="text-sm font-medium text-white">{u.username}</span>
                                                <span className="text-xs text-gray-400">{formatSize(u.storage_used)} / {formatSize(u.storage_quota)}</span>
                                            </div>
                                            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                                                <div className="h-full bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full transition-all"
                                                    style={{ width: `${Math.min(100, (u.storage_used / (u.storage_quota || 1)) * 100)}%` }} />
                                            </div>
                                            <div className="text-[10px] text-gray-600 mt-1">{u.file_count} files</div>
                                        </div>
                                    ))}
                                    {storageData.length === 0 && <div className="text-gray-600 text-sm">No storage data.</div>}
                                </div>
                            )}
                            {tab === 'system' && (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="text-xs text-zinc-500">Service state refreshes every 15 seconds.</div>
                                        <button onClick={loadSystem} className="px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.03] text-xs text-zinc-400 hover:text-white hover:bg-white/[0.06] transition-colors">
                                            Refresh now
                                        </button>
                                    </div>
                                    {!systemInfo && <div className="text-gray-500 text-sm">Loading system info…</div>}
                                    {systemInfo && (<>
                                        <div className="grid grid-cols-2 gap-3">
                                            {Object.entries(systemInfo.services || {}).map(([name, status]) => {
                                                const readiness = systemInfo.readiness?.[name];
                                                const observedAt = readiness?.verified_at || readiness?.observed_at;
                                                const health = getServiceHealthPresentation(status, readiness);
                                                const isBusy = health.tone === 'warning';
                                                const isHealthy = health.tone === 'healthy';
                                                const dotClass = isBusy ? 'bg-amber-400' : isHealthy ? 'bg-green-400' : 'bg-red-400';
                                                const statusClass = isBusy ? 'text-amber-400' : isHealthy ? 'text-green-500' : 'text-red-400';
                                                const readinessLabel = health.readinessState
                                                    ? health.readinessState.replace(/_/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())
                                                    : null;
                                                return (
                                                <div key={name} data-testid={`system-service-${name}`} data-health-tone={health.tone} className="p-4 bg-white/[0.03] border border-white/5 rounded-xl flex items-start gap-3">
                                                    <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 mt-1 ${dotClass}`} />
                                                    <div className="min-w-0 flex-1">
                                                        <div className="flex flex-wrap items-center gap-2">
                                                            <div className="text-sm font-medium text-white capitalize">{name.replace(/_/g, ' ')}</div>
                                                            {readiness?.topology === 'private' && (
                                                                <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border border-zinc-600/40 bg-zinc-600/10 text-zinc-400">Private network</span>
                                                            )}
                                                        </div>
                                                        <div className={`text-xs ${statusClass}`}>{health.label}</div>
                                                        {readinessLabel && !isBusy && <div className="text-[10px] text-zinc-500 mt-1">Readiness: {readinessLabel}</div>}
                                                        {observedAt && <div className="text-[10px] text-zinc-600 mt-0.5" title={observedAt}>Verified {new Date(observedAt).toLocaleString()}</div>}
                                                        {readiness?.detail && <div className={`text-[10px] mt-1 break-words ${isBusy ? 'text-amber-400/80' : isHealthy ? 'text-zinc-500' : 'text-red-400/80'}`}>{readiness.detail}</div>}
                                                    </div>
                                                </div>
                                            )})}
                                        </div>
                                        <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">App Version</div>
                                            <div className="text-white font-mono text-sm">{systemInfo.version}</div>
                                        </div>
                                        {systemInfo.index && (
                                            <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">PostgreSQL Vector Index</div>
                                                <div className="text-2xl font-bold text-white">{(systemInfo.index.vectors || 0).toLocaleString()}</div>
                                                <div className="text-xs text-gray-600 mt-1">vectors · {systemInfo.index.queued_jobs || 0} queued · {systemInfo.index.active_jobs || 0} active</div>
                                                <button
                                                    onClick={async () => {
                                                        if (!confirm('Delete ALL embeddings? Files must be re-indexed to become searchable again. This cannot be undone.')) return;
                                                        setClearProcessing(true);
                                                        try {
                                                            await api.adminClearEmbeddings();
                                                            loadSystem();
                                                        } catch (e) {
                                                            alert('Failed: ' + e.message);
                                                        } finally {
                                                            setClearProcessing(false);
                                                        }
                                                    }}
                                                    disabled={clearProcessing}
                                                    className="mt-3 w-full bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-500/20 py-2 rounded-lg text-xs font-medium transition-all disabled:opacity-50"
                                                >
                                                    {clearProcessing ? 'Clearing...' : 'Clear All Embeddings'}
                                                </button>
                                            </div>
                                        )}
                                        <div className="p-4 bg-white/[0.03] border border-white/5 rounded-xl">
                                            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Total Users</div>
                                            <div className="text-2xl font-bold text-white">{systemInfo.user_count}</div>
                                        </div>
                                    </>)}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            );
        }

        // ── USERNAME SETUP MODAL (first Google login) ────────────────────────────
        function UsernameSetupModal({ currentUsername, token, onConfirm }) {
            const [value, setValue] = React.useState(currentUsername || '');
            const [saving, setSaving] = React.useState(false);
            const [error, setError] = React.useState('');

            const handleSave = async () => {
                setSaving(true);
                setError('');
                try {
                    const res = await fetch('/api/auth/username', {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                        body: JSON.stringify({ username: value.trim() })
                    });
                    if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Failed to save username'); }
                    localStorage.setItem('username', value.trim());
                    onConfirm(value.trim());
                } catch(e) { setError(e.message); }
                finally { setSaving(false); }
            };

            return (
                <div className="fixed inset-0 z-[500] flex items-center justify-center bg-black/80 backdrop-blur-sm">
                    <div className="bg-[#111113] border border-white/10 rounded-2xl p-8 w-full max-w-sm shadow-2xl">
                        <div className="w-12 h-12 rounded-full bg-indigo-600/20 flex items-center justify-center mb-4">
                            <svg className="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                        </div>
                        <h2 className="text-lg font-bold text-white mb-1">Set your username</h2>
                        <p className="text-xs text-zinc-400 mb-5">
                            We've generated a username from your Google account. You can keep it or choose a new one.
                        </p>
                        <input
                            value={value}
                            onChange={e => setValue(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && !saving && value.trim() && handleSave()}
                            className="w-full bg-zinc-900 border border-zinc-700 focus:border-indigo-500 rounded-xl px-4 py-3 text-sm text-white outline-none mb-3"
                            placeholder="your-username"
                            autoFocus
                        />
                        {error && <p className="text-red-400 text-xs mb-3">{error}</p>}
                        <button onClick={handleSave} disabled={saving || !value.trim()}
                            className="w-full bg-indigo-600/15 border border-indigo-500/30 hover:bg-indigo-600/25 disabled:opacity-50 text-indigo-200 hover:text-white font-bold py-3 rounded-xl text-sm transition-all">
                            {saving ? 'Saving…' : 'Confirm Username'}
                        </button>
                        <button onClick={() => onConfirm(currentUsername)}
                            className="w-full mt-2 text-zinc-500 hover:text-zinc-300 text-xs py-2 transition-colors">
                            Keep "{currentUsername}"
                        </button>
                    </div>
                </div>
            );
        }

        // ── ERROR PAGES ─────────────────────────────────────────────────────────
        function AppCrashPage({ error, onRetry }) {
            return (
                <div className="fixed inset-0 bg-black flex items-center justify-center p-4">
                    <div className="w-full max-w-lg bg-[#050505] border border-[#1a1a1a] rounded-3xl overflow-hidden shadow-2xl">
                        <div className="h-1 bg-gradient-to-r from-red-600 via-orange-500 to-red-600 animate-pulse"></div>
                        <div className="p-10 text-center">
                            <div className="w-16 h-16 rounded-2xl bg-red-500/10 flex items-center justify-center mx-auto mb-6">
                                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" className="text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                            <p className="text-xs font-semibold tracking-widest uppercase text-red-500 mb-2">Application Error</p>
                            <h1 className="text-2xl font-bold text-white mb-3">Something went wrong</h1>
                            <p className="text-zinc-400 text-sm mb-6">An unexpected error occurred in the application. Reloading usually fixes this.</p>
                            {error && (
                                <pre className="text-left text-xs text-zinc-500 bg-black/50 border border-white/5 rounded-xl p-4 mb-6 overflow-auto max-h-32 font-mono">
                                    {error.message || String(error)}
                                </pre>
                            )}
                            <button
                                onClick={onRetry}
                                className="px-8 py-3 bg-red-600 hover:bg-red-500 text-white text-sm font-semibold uppercase tracking-wide rounded-xl transition-colors"
                            >
                                Reload Page
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        function BackendErrorPage({ onRetry, memory }) {
            const [countdown, setCountdown] = React.useState(30);
            useEffect(() => {
                const id = setInterval(() => setCountdown(c => {
                    if (c <= 1) { clearInterval(id); onRetry(); return 0; }
                    return c - 1;
                }), 1000);
                return () => clearInterval(id);
            }, []);
            return (
                <div className="fixed inset-0 bg-black flex items-center justify-center p-4">
                    <div className="w-full max-w-md bg-[#050505] border border-[#1a1a1a] rounded-3xl overflow-hidden shadow-2xl">
                        <div className="h-1 bg-gradient-to-r from-red-600 via-orange-500 to-red-600 animate-pulse"></div>
                        <div className="p-10 text-center">
                            <div className="w-16 h-16 rounded-2xl bg-red-500/10 flex items-center justify-center mx-auto mb-6">
                                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" className="text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                                </svg>
                            </div>
                            <p className="text-xs font-semibold tracking-widest uppercase text-red-500 mb-2">Server Error</p>
                            <h1 className="text-2xl font-bold text-white mb-3">Backend Not Responding</h1>
                            <p className="text-zinc-400 text-sm mb-6">
                                Available RAM is below 7 percent. Please wait while memory is freed.
                            </p>
                            {memory && (
                                <p className="text-zinc-600 text-xs mb-6">
                                    RAM available: <span className="text-zinc-400 font-semibold">{memory.available_percent}%</span>
                                </p>
                            )}
                            <p className="text-zinc-600 text-xs mb-6">Retrying automatically in <span className="text-zinc-400 font-semibold">{countdown}s</span></p>
                            <button
                                onClick={onRetry}
                                className="px-8 py-3 bg-red-600 hover:bg-red-500 text-white text-sm font-semibold uppercase tracking-wide rounded-xl transition-colors"
                            >
                                Retry Now
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        function ConnectionErrorPage({ onRetry }) {
            return (
                <div className="fixed inset-0 bg-black flex items-center justify-center p-4">
                    <div className="w-full max-w-md bg-[#050505] border border-[#1a1a1a] rounded-3xl overflow-hidden shadow-2xl">
                        <div className="h-1 bg-gradient-to-r from-zinc-700 via-zinc-500 to-zinc-700"></div>
                        <div className="p-10 text-center">
                            <div className="w-16 h-16 rounded-2xl bg-zinc-500/10 flex items-center justify-center mx-auto mb-6">
                                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" className="text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m-2.829-2.829a5 5 0 000-7.07M6.343 6.343a8 8 0 000 11.314" />
                                </svg>
                            </div>
                            <p className="text-xs font-semibold tracking-widest uppercase text-zinc-500 mb-2">Connection Issue</p>
                            <h1 className="text-2xl font-bold text-white mb-3">Reconnecting</h1>
                            <p className="text-zinc-400 text-sm mb-6">
                                Lavix Vault could not complete the health check. Try again after the server is reachable.
                            </p>
                            <button
                                onClick={onRetry}
                                className="px-8 py-3 bg-zinc-800 hover:bg-zinc-700 text-white text-sm font-semibold uppercase tracking-wide rounded-xl transition-colors"
                            >
                                Retry Now
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        function CheckingPage() {
            return (
                <div className="fixed inset-0 bg-black flex items-center justify-center">
                    <div className="text-center">
                        <div className="w-10 h-10 border-2 border-white/10 border-t-red-500 rounded-full animate-spin mx-auto mb-4"></div>
                        <p className="text-zinc-500 text-sm">Connecting to Lavix Vault…</p>
                    </div>
                </div>
            );
        }

        // ── ERROR BOUNDARY ────────────────────────────────────────────────────────
        class ErrorBoundary extends Component {
            constructor(props) {
                super(props);
                this.state = { hasError: false, error: null };
            }
            static getDerivedStateFromError(error) {
                return { hasError: true, error };
            }
            componentDidCatch(error, info) {
                console.error('[Lavix Vault] React crash:', error, info.componentStack);
            }
            render() {
                if (this.state.hasError) {
                    return <AppCrashPage error={this.state.error} onRetry={() => window.location.reload()} />;
                }
                return this.props.children;
            }
        }

        // ── APP (ROOT) ───────────────────────────────────────────────────────────
        function App() {
            const [token, setToken] = useState(localStorage.getItem('token'));
            const [user, setUser] = useState(localStorage.getItem('username'));
            const [userProfile, setUserProfile] = useState(null);
            const [oauthExchangeError, setOauthExchangeError] = useState(null);
            const [showUsernamePrompt, setShowUsernamePrompt] = useState(false);
            const [chatPreTagIds, setChatPreTagIds] = useState([]);
            const [view, setView] = useState('files'); // files, upload, chat, settings, admin
            const [currentTab, setCurrentTab] = useState('files'); // files, trash
            const [files, setFiles] = useState([]);
            const [previewFile, setPreviewFile] = useState(null);
            const [fileSearch, setFileSearch] = useState(localStorage.getItem('fileSearch') !== 'false');
            const [showDSConfirm, setShowDSConfirm] = useState(false);
            const [confirmState, setConfirmState] = useState(null);
            const [sessionExpired, setSessionExpired] = useState(false);
            const [sessionExpiredDetail, setSessionExpiredDetail] = useState(null);
            const [uploadPaused, setUploadPaused] = useState(false);
            const [indexingActive, setIndexingActive] = useState(false);
            const autoCloseRef = useRef(false);
            const [backendStatus, setBackendStatus] = useState('checking'); // 'checking' | 'ok' | 'low_memory' | 'unreachable'
            const [backendMemory, setBackendMemory] = useState(null);
            const [searchTerm, setSearchTerm] = useState('');
            const [prefFontSize, setPrefFontSize] = useState(localStorage.getItem('fontSize') || 'md');
            const [prefSendOnEnter, setPrefSendOnEnter] = useState(localStorage.getItem('sendOnEnter') !== 'false');
            const [prefTimestamps, setPrefTimestamps] = useState(localStorage.getItem('showTimestamps') === 'true');
            const [prefAutoScroll, setPrefAutoScroll] = useState(localStorage.getItem('autoScroll') !== 'false');

            useEffect(() => {
                const size = localStorage.getItem('fontSize') || 'md';
                document.documentElement.style.setProperty('--chat-font-size', size === 'sm' ? '13px' : size === 'lg' ? '17px' : '15px');
            }, []);

            // Backend health check on mount
            useEffect(() => {
                // ── Handle Google OAuth redirect ──────────────────────────────
                const _up = new URLSearchParams(window.location.search);
                const _gk = _up.get('google_auth');
                const _ge = _up.get('google_error');
                if (_gk || _ge) {
                    window.history.replaceState({}, '', window.location.pathname);
                    if (_gk) {
                        fetch(`/api/auth/google/token?key=${encodeURIComponent(_gk)}`)
                            .then(r => {
                                return r.ok ? r.json() : r.text().then(t => Promise.reject(t));
                            })
                            .then(resp => {
                                storeAuthSession(resp, resp.username);
                                if (resp.username) {
                                    setUser(resp.username);
                                }
                                setToken(resp.access_token);
                                if (resp.is_new_user) setShowUsernamePrompt(true);
                            })
                            .catch(e => {
                                console.error('[GoogleAuth] FAILED:', e);
                                setOauthExchangeError('Google sign-in failed: ' + e);
                            });
                    } else {
                        const _em = { cancelled: 'Google sign-in was cancelled.', invalid_state: 'Security error — please try again.', token_exchange_failed: 'Could not complete Google sign-in.', account_inactive: 'Account is inactive.', invalid_token: 'Google token verification failed.' };
                        setOauthExchangeError(_em[_ge] || 'Google sign-in failed.');
                    }
                }
                // ── Health check ─────────────────────────────────────────────
                let cancelled = false;
                const applyHealth = async (res) => {
                    let data = {};
                    try { data = await res.json(); } catch(e) {}
                    if (cancelled) return;
                    setBackendMemory(data.memory || null);
                    setBackendStatus(data.reason === 'low_memory' ? 'low_memory' : (res.ok ? 'ok' : 'unreachable'));
                };
                const check = () => {
                    fetch('/api/health', { signal: AbortSignal.timeout(5000) })
                        .then(applyHealth)
                        .catch(() => { if (!cancelled) setBackendStatus('unreachable'); });
                };
                check();
                // Listen for 5xx events dispatched by vault-client.js
                const onBackendError = (event) => {
                    if (event.detail?.reason === 'low_memory') setBackendStatus('low_memory');
                };
                window.addEventListener('backend-error', onBackendError);
                return () => { cancelled = true; window.removeEventListener('backend-error', onBackendError); };
            }, []);

            // Re-check when user asks to retry
            useEffect(() => {
                if (backendStatus !== 'checking') return;
                fetch('/api/health', { signal: AbortSignal.timeout(5000) })
                    .then(async r => {
                        let data = {};
                        try { data = await r.json(); } catch(e) {}
                        setBackendMemory(data.memory || null);
                        setBackendStatus(data.reason === 'low_memory' ? 'low_memory' : (r.ok ? 'ok' : 'unreachable'));
                    })
                    .catch(() => setBackendStatus('unreachable'));
            }, [backendStatus]);


            // Responsive Sidebar State
            const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
            const [isMobileOpen, setIsMobileOpen] = useState(false);

            // Details panel state (lifted from Dashboard — works for files AND folders)
            const [selectedItem, setSelectedItem] = useState(null);
            const [filesLoading, setFilesLoading] = useState(true);

            // Sync the current indexing state into the details panel when files refresh
            useEffect(() => {
                if (selectedItem && selectedItem._type !== 'folder') {
                    const updated = files.find(f => f.id === selectedItem.id);
                    if (updated && getIndexState(updated) !== getIndexState(selectedItem)) {
                        setSelectedItem(updated);
                    }
                }
            }, [files]);

            const handleGrantAccessForPanel = async (fileId) => {
                try {
                    const result = await api.grantAIAccess(fileId);
                    const state = result.state || result.status || 'queued';
                    setSelectedItem(prev => prev && prev.id === fileId ? { ...prev, ai_status: state, ingestion_state: state } : prev);
                    await refreshFiles();
                    await refreshStats();
                } catch (e) {
                    console.error('Grant access failed:', e);
                    // Never silent: the button otherwise just flips back with
                    // no explanation of why indexing didn't start.
                    showToast(describeIndexingError(e, selectedItem?.filename || 'File'), 'info');
                }
            };

            const revokeAccessForPanel = async (fileId, close) => {
                try {
                    const result = await api.revokeAIAccess(fileId);
                    const state = result.state || result.status || 'not_granted';
                    setSelectedItem(prev => prev && prev.id === fileId ? { ...prev, ai_status: state, ingestion_state: state } : prev);
                    if (typeof close === 'function') close();
                    refreshFiles().catch((e) => {
                        console.error('Revoke refresh failed:', e);
                        showToast(describeIndexingError(e, selectedItem?.filename || 'File'), 'info');
                    });
                    refreshStats().catch((e) => {
                        console.error('Revoke refresh failed:', e);
                    });
                } catch (e) {
                    console.error('Revoke access failed:', e);
                    showToast(describeIndexingError(e, selectedItem?.filename || 'File'), 'info');
                }
            };

            const handleRevokeAccessForPanel = (fileId) => {
                const target = files.find(file => file.id === fileId) || selectedItem;
                onConfirmAction({
                    title: 'Revoke AI access?',
                    message: `This removes generated summaries, tags, embeddings, and searchable revisions for ${target?.filename || 'this file'}. The original file is preserved, but it must be indexed again before AI search can use it.`,
                    confirmText: 'Revoke access',
                    variant: 'danger',
                    onConfirm: (close) => revokeAccessForPanel(fileId, close),
                });
            };

            const handleCancelIndexingForPanel = async (fileId) => {
                try {
                    await api.cancelIndexing(fileId);
                    await Promise.all([refreshFiles(), refreshStats()]);
                } catch (e) { console.error('Cancel indexing failed:', e); }
            };

            const toggleFileSearch = () => {
                const next = !fileSearch;
                setFileSearch(next);
                localStorage.setItem('fileSearch', String(next));
            };

            const handleAiReadyClick = async () => {
                await refreshStats();
                // A previous completed run arms the 2s auto-close timer;
                // opening the widget manually must always clear it so the
                // dialog stays open until Activate/Cancel.
                autoCloseRef.current = false;
                setShowDSConfirm(true);
            };

            const cancelIndexing = async () => {
                setIndexingActive(false);
                setShowDSConfirm(false);
                try {
                    await api.cancelIndexing();
                } catch(e) {
                    console.error('Failed to cancel indexing:', e);
                }
                await Promise.all([refreshFiles(), refreshStats()]);
            };

            const confirmDeepSearch = async () => {
                setIndexingActive(true);
                setFileSearch(true);
                localStorage.setItem('fileSearch', 'true');
                try {
                    await api.enableAllEmbeddings();
                    // This run was launched here: arm the 2s auto-close.
                    autoCloseRef.current = true;
                    await refreshStats();
                    await refreshFiles();
                    setShowDSConfirm(false);
                } catch (e) {
                    console.error('Failed to enable file indexing:', e);
                    // Failed launch must not auto-close: leave the dialog
                    // open on the error instead of vanishing after 2s.
                    autoCloseRef.current = false;
                    setIndexingActive(false);
                }
            };

            // Chat Sessions (Redis - for LLM context)
            const [sessions, setSessions] = useState([]);
            const [currentSessionId, setCurrentSessionId] = useState(null);
            // Persistent chat (PostgreSQL)
            const [currentPgChatId, setCurrentPgChatId] = useState(null);
            const newChatIntentRef = useRef(false);
            // Pinned / favourite sessions (localStorage-persisted)
            const [pinnedSessions, setPinnedSessions] = useState(() => {
                try { return JSON.parse(localStorage.getItem('vault_pinned_sessions') || '[]'); }
                catch { return []; }
            });
            const togglePinSession = (sid) => {
                setPinnedSessions(prev => {
                    const next = prev.includes(sid) ? prev.filter(x => x !== sid) : [sid, ...prev];
                    localStorage.setItem('vault_pinned_sessions', JSON.stringify(next));
                    return next;
                });
            };
            // PostgreSQL persistent chats
            const [pgChats, setPgChats] = useState([]);

            // Index statistics for searchable files
            const [aiStats, setAiStats] = useState(null);

            // Storage Stats
            const [totalSizeBytes, setTotalSizeBytes] = useState(0);
            const [storageQuotaBytes, setStorageQuotaBytes] = useState(20 * 1024 * 1024 * 1024); // Initial default
            const [appVersion, setAppVersion] = useState('');
            const refreshFilesRef = useRef(() => {});


            // Folders list (synced from Dashboard for upload targeting)
            const [appFolders, setAppFolders] = useState([]);

            // Upload State (Lifted for Background Uploads)
            const [uploadFiles, setUploadFiles] = useState([]);
            const [uploading, setUploading] = useState(false);
            const [uploadTargetFolder, setUploadTargetFolder] = useState(null); // null = root
            const [minimized, setMinimized] = useState(false);
            const [uploadProgress, setUploadProgress] = useState(0);
            const [uploadCurrentIndex, setUploadCurrentIndex] = useState(0);
            const [uploadResults, setUploadResults] = useState([]);
            const [uploadShowResults, setUploadShowResults] = useState(false);
            const [uploadStartTime, setUploadStartTime] = useState(null);
            const [uploadElapsedTime, setUploadElapsedTime] = useState(0);
            const uploadAbortRef = useRef(null);
            const sessionSuspendedRef = useRef(false);

            // Upload Timer
            useEffect(() => {
                let interval;
                if (uploading && uploadStartTime) {
                    interval = setInterval(() => {
                        setUploadElapsedTime(Math.floor((Date.now() - uploadStartTime) / 1000));
                    }, 500);
                }
                return () => clearInterval(interval);
            }, [uploading, uploadStartTime]);

            useEffect(() => {
                if (token && !sessionExpired) {
                    refreshSessions();
                    api.me().then(p => setUserProfile(p)).catch(() => {});
                }
                const handleAuthError = (event) => {
                    sessionSuspendedRef.current = true;
                    const hadActiveUpload = Boolean(uploadAbortRef.current);
                    if (uploadAbortRef.current) {
                        uploadAbortRef.current.abort();
                        uploadAbortRef.current = null;
                    }
                    if (hadActiveUpload) {
                        setUploadPaused(true);
                        setUploading(false);
                    }
                    setSessionExpiredDetail(event.detail || null);
                    setSessionExpired(true);
                    window.dispatchEvent(new CustomEvent('session-suspended', { detail: event.detail || null }));
                };
                const handleAuthRefreshed = (event) => {
                    if (event.detail?.access_token) setToken(event.detail.access_token);
                };
                window.addEventListener('auth-error', handleAuthError);
                window.addEventListener('auth-refreshed', handleAuthRefreshed);
                return () => {
                    window.removeEventListener('auth-error', handleAuthError);
                    window.removeEventListener('auth-refreshed', handleAuthRefreshed);
                };
            }, [token, sessionExpired]);

            useEffect(() => {
                if (!token || sessionExpired) return undefined;
                const activity = installSessionActivityTracking();
                return () => activity.dispose();
            }, [token, sessionExpired]);

            // ── Indexing: tracking & effects ──────────────────────────────
            const isIndexing = indexingActive && fileSearch;

            // Auto-resolve: when stats show all done while indexing
            useEffect(() => {
                if (indexingActive && aiStats) {
                    const progress = getIndexProgress(aiStats);
                    const terminal = progress.searchable
                        + (aiStats.failed || 0)
                        + (aiStats.cancelled || 0)
                        + (aiStats.password_required || 0);
                    if (terminal >= progress.indexable) {
                        setIndexingActive(false);
                    }
                }
            }, [indexingActive, aiStats]);

            // Auto-close dialog 2s after indexing finishes — but only for
            // runs launched from this dialog's own ACTIVATE button (armed
            // there on success). Background runs must never arm it, or a
            // manually opened dialog closes mid-read when unrelated work
            // finishes. Manual opens always clear the flag (see
            // handleAiReadyClick), so the dialog stays until Activate/Cancel.
            useEffect(() => {
                if (showDSConfirm && !indexingActive && autoCloseRef.current) {
                    const timer = setTimeout(() => setShowDSConfirm(false), 2000);
                    return () => clearTimeout(timer);
                }
            }, [showDSConfirm, indexingActive]);

            // Polling: refresh stats every 8s while indexing
            useEffect(() => {
                let interval;
                if (indexingActive && token && !sessionExpired) {
                    interval = setInterval(() => {
                        refreshStats();
                        refreshFiles({ showLoading: false });
                    }, 8000);
                }
                return () => clearInterval(interval);
            }, [indexingActive, token, sessionExpired]);

            // Dialog-open polling: the Enable/indexing dialog must stay
            // live while open even when nothing is flagged active (freshly
            // granted files not yet picked up, states settling after bulk
            // actions). Without this the dialog freezes at open-time values
            // while the rest of the UI moves on.
            useEffect(() => {
                if (!showDSConfirm || !token || sessionExpired) return undefined;
                const interval = setInterval(() => {
                    refreshStats();
                    refreshFiles({ showLoading: false });
                }, 8000);
                return () => clearInterval(interval);
            }, [showDSConfirm, token, sessionExpired]);

            // Auto-Resume Upload Effect
            useEffect(() => {
                if (!sessionExpired && uploadPaused && uploadFiles.length > 0) {
                    const timer = setTimeout(() => {
                        upload(uploadFiles, uploadCurrentIndex);
                    }, 1000);
                    return () => clearTimeout(timer);
                }
            }, [sessionExpired, uploadPaused]);

            const onConfirmAction = (config) => {
                setConfirmState({
                    ...config,
                    onConfirm: async () => {
                        // Handlers may close the modal early (e.g. right
                        // after their POST resolves) and let refresh work
                        // continue in the background. Handlers that never
                        // call close() keep the legacy behavior — and a
                        // throwing handler can no longer wedge the modal
                        // open forever.
                        let closed = false;
                        const close = () => { closed = true; setConfirmState(null); };
                        try {
                            await config.onConfirm(close);
                        } finally {
                            if (!closed) setConfirmState(null);
                        }
                    }
                });
            };

            const refreshSessions = async () => {
                try {
                    const data = await api.getChatSessions();
                    setSessions(data.sessions || []);
                    if (!currentSessionId && data.sessions && data.sessions.length > 0) {
                        setCurrentSessionId(data.sessions[0].id);
                    } else if (!currentSessionId) {
                        const defaultSid = `user_${user}_default`;
                        setCurrentSessionId(defaultSid);
                    }
                } catch (e) {
                    console.error('Failed to load sessions:', e);
                }
                // Also load PG chats
                try {
                    const pgData = await api.listChats();
                    setPgChats(pgData.chats || []);
                    // Never auto-select a PG chat on refresh. The old logic
                    // picked pgData.chats[0] which was always the pinned chat
                    // (API sorts pinned DESC), hijacking the user's view from
                    // their current/new chat to the favourite. Let the user
                    // explicitly select from the sidebar instead.
                    newChatIntentRef.current = false;
                } catch(e) {
                    console.error('Failed to load pg chats:', e);
                }
            };

            const refreshStats = async () => {
                if (!token) return;
                try {
                    const data = await api.getAiStats();
                    setAiStats(data);
                } catch (e) {
                    console.error('Failed to load AI stats:', e);
                }
            };

            const handleNewChat = () => {
                newChatIntentRef.current = true;
                const newSid = `user_${user}_chat_${Date.now()}`;
                setCurrentSessionId(newSid);
                setCurrentPgChatId(null);
                setView('chat');
                // Chat record is created in sendRaw when the first message is sent
            };

            // "Chat about this file" from the file details panel: always a NEW
            // chat (never silently mutate the current chat's scope). The file
            // id rides chatPreTagIds — the established dashboard-to-chat path
            // (ChatView seeds taggedFiles from it, scope commits on send).
            // NOTE: setTaggedFiles lives in ChatView scope and is NOT visible
            // here; calling it throws and the click silently does nothing.
            const handleChatAboutFile = (f) => {
                const fid = Number(f?.file_id ?? f?.id);
                if (!fid) return;
                newChatIntentRef.current = true;
                setCurrentSessionId(`user_${user}_chat_${Date.now()}`);
                setCurrentPgChatId(null);
                setChatPreTagIds([fid]);
                setSelectedItem(null);
                setView('chat');
            };

            const handleOpenPgChat = async (chatId) => {
                setCurrentPgChatId(chatId);
                setCurrentSessionId(`user_${user}_pgchat_${chatId}`);
                setView('chat');
            };

            const handlePgChatCreated = (chat) => {
                setCurrentPgChatId(chat.id);
                setPgChats(prev => [chat, ...prev.filter(c => c.id !== chat.id)]);
            };

            const handleDeleteSession = async (sid) => {
                onConfirmAction({
                    title: 'Delete Chat Session?',
                    message: 'All messages and context for this session will be wiped.',
                    variant: 'danger',
                    onConfirm: async () => {
                        await api.deleteChatSession(sid);
                        if (currentSessionId === sid) {
                            setCurrentSessionId(null);
                        }
                        await refreshSessions();
                    }
                });
            };

            const handleDeletePgChat = async (chatId) => {
                onConfirmAction({
                    title: 'Delete Chat?',
                    message: 'This chat and all its messages will be permanently deleted.',
                    variant: 'danger',
                    onConfirm: async () => {
                        await api.deleteChat(chatId);
                        setPgChats(prev => prev.filter(c => c.id !== chatId));
                    }
                });
            };

            const handlePinChat = async (chatId, currentPinned) => {
                try {
                    await api.patchChat(chatId, { pinned: !currentPinned });
                    setPgChats(prev => prev.map(c => c.id === chatId ? { ...c, pinned: !currentPinned } : c));
                } catch(e) {
                    console.error('Failed to pin chat:', e);
                }
            };

            const refreshFiles = async ({ showLoading = true } = {}) => {
                if (showLoading) setFilesLoading(true);
                try {
                    // Only show deleted files when explicitly on the Trash tab in Dashboard.
                    // In Chat view, always load active files so @mention works correctly.
                    const data = await api.getFiles(view === 'files' && currentTab === 'trash');
                    const loadedFiles = data?.files || [];

                    // Normalize and Debug
                    const normalized = loadedFiles.map(f => ({
                        ...f,
                        is_deleted: f.is_deleted !== undefined ? f.is_deleted : false
                    }));

                    setFiles(normalized);
                    setTotalSizeBytes(data.total_size_bytes || 0);
                    setStorageQuotaBytes(data.storage_quota_bytes || (50 * 1024 * 1024 * 1024));
                    setAppVersion(data.version || '0.0.1');
                    refreshStats();

                    // Restore indexing state after page reload: if any file is still processing,
                    // indexing was active before the reload.
                    if (normalized.some(isIndexActive)) {
                        setIndexingActive(true);
                    }
                } catch (e) {
                    console.error('Failed to load files:', e);
                    if (showLoading) setFiles([]);
                } finally {
                    if (showLoading) setFilesLoading(false);
                }
            };

            useEffect(() => {
                if (token) {
                    refreshFiles();
                }
            }, [token, view, currentTab]);

            // Admin quota changes fire this event (the admin panel lives under
            // the Settings overlay, so no view switch refreshes the STORAGE box).
            useEffect(() => {
                refreshFilesRef.current = refreshFiles;
                const onStorageChanged = () => { refreshFilesRef.current({ showLoading: false }); };
                window.addEventListener('storage-changed', onStorageChanged);
                return () => window.removeEventListener('storage-changed', onStorageChanged);
            }, [refreshFiles]);

            // Auto-refresh processing files
            useEffect(() => {
                if (!files.some(isIndexActive)) return;

                const interval = setInterval(() => {
                    refreshFiles({ showLoading: false });
                    refreshStats();
                }, 10000);

                return () => clearInterval(interval);
            }, [files, token]);

            const handleDismissUploadResults = () => {
                setUploadShowResults(false);
                setUploadFiles([]);
                setUploadResults([]);
                setUploadProgress(0);
                setUploadStartTime(null);
                setUploadElapsedTime(0);
                // Refresh files as upload is done
                refreshFiles();
                if (view === 'upload') {
                    setView('files');
                }
            };

            const estimatedRemaining = () => {
                if (uploadCurrentIndex === 0 || uploadElapsedTime === 0) return '...';
                const avgPerFile = uploadElapsedTime / (uploadCurrentIndex + (uploadProgress / 100));
                const remaining = Math.round(avgPerFile * (uploadFiles.length - uploadCurrentIndex - 1));

                if (remaining < 1) return '<1s';
                if (remaining < 60) return `${remaining}s`;
                const mins = Math.floor(remaining / 60);
                const secs = remaining % 60;
                return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
            };

            const upload = async (filesToUpload = uploadFiles, startIndex = 0, overrideFolderId = undefined, folderIdPerFile = null) => {
                if (!filesToUpload.length) return;

                const controller = new AbortController();
                uploadAbortRef.current = controller;
                setUploading(true);
                setUploadPaused(false);

                if (startIndex === 0) {
                    setUploadCurrentIndex(0);
                    setUploadProgress(0);
                    setUploadResults([]);
                    setUploadShowResults(false);
                    setUploadStartTime(Date.now());
                    setUploadElapsedTime(0);
                }

                // Preserve results if resuming
                const results = startIndex === 0 ? [] : [...uploadResults];

                for (let i = startIndex; i < filesToUpload.length; i++) {
                    setUploadCurrentIndex(i);
                    setUploadProgress(10);

                    const interval = setInterval(() => {
                        setUploadProgress(prev => Math.min(prev + 3, 95));
                    }, 500);

                    try {
                        const folderId = folderIdPerFile ? folderIdPerFile[i] : (overrideFolderId !== undefined ? overrideFolderId : uploadTargetFolder);
                        const data = await api.uploadFile(filesToUpload[i], false, folderId, controller.signal);
                        results.push({ name: filesToUpload[i].name, success: true, id: data.file_id });
                        setUploadResults([...results]);
                    } catch (e) {
                        if (sessionSuspendedRef.current || e.message === 'SessionExpired') {
                            clearInterval(interval);
                            if (uploadAbortRef.current === controller) uploadAbortRef.current = null;
                            setUploadPaused(true);
                            setUploading(false);
                            return; // Pause loop
                        }

                        console.error(`Upload failed for ${filesToUpload[i].name}:`, e);
                        results.push({
                            name: filesToUpload[i].name,
                            success: false,
                            error: e.message,
                            existing_id: e.data?.existing_id,
                            existing_mime: e.data?.mime_type,
                            file_obj: filesToUpload[i]
                        });
                        setUploadResults([...results]);
                    }

                    clearInterval(interval);
                    setUploadProgress(100);
                    await new Promise(r => setTimeout(r, 200));
                }

                if (uploadAbortRef.current === controller) uploadAbortRef.current = null;
                setUploadResults(results);
                setUploading(false);

                // Refresh folder counts so grid cards show updated file counts
                const foldersData = await api.getFolders().catch(() => []);
                setAppFolders(foldersData);

                const failedCount = results.filter(r => !r.success).length;
                if (failedCount > 0) {
                    setUploadShowResults(true);
                    if (minimized) {
                        setMinimized(false);
                        setView('upload');
                    }
                } else {
                    setTimeout(async () => {
                        if (!minimized) {
                            handleDismissUploadResults();
                        } else {
                            setUploadFiles([]);
                        }
                        // Every completed upload refreshes the list, not
                        // just minimized ones — otherwise new files stay
                        // invisible until a manual refresh or tab switch.
                        refreshFiles();
                        // New uploads change the eligible-files denominator
                        // even before AI access is granted — refresh stats so
                        // the AI-READY count moves immediately.
                        refreshStats();
                        // Files uploaded via Dashboard are NOT auto-indexed.
                        // Indexing only happens when files are uploaded from the chat box
                        // or when the user explicitly grants AI access.
                    }, 800);
                }
            };

            const handleUploadFiles = (newFiles, overrideFolderId = undefined) => {
                const updated = Array.from(newFiles);
                setUploadFiles(updated);
                setUploadResults([]);
                setUploadShowResults(false);
                if (updated.length > 0) {
                    setTimeout(() => upload(updated, 0, overrideFolderId), 100);
                }
            };

            // Upload a list of {file, folderId} pairs in a single sequential pass.
            // This avoids the race condition caused by calling handleUploadFiles multiple
            // times (once per sub-folder), which overwrites state and launches concurrent uploads.
            const handleUploadPairs = (pairs) => {
                if (!pairs.length) return;
                const files = pairs.map(p => p.file);
                const folderIds = pairs.map(p => p.folderId);
                setUploadFiles(files);
                setUploadResults([]);
                setUploadShowResults(false);
                setTimeout(() => upload(files, 0, undefined, folderIds), 100);
            };

            const handleReplaceFile = async (fileObj) => {
                try {

                    // Find index to show correct name in mini-widget
                    const fidx = uploadFiles.findIndex(f => f.name === fileObj.name);
                    if (fidx !== -1) setUploadCurrentIndex(fidx);

                    // Minimize instead of showing full screen circle
                    setMinimized(true);
                    setView('files');

                    setUploading(true);
                    setUploadProgress(20);

                    const data = await api.uploadFile(fileObj, true);

                    setUploadProgress(100);
                    refreshFiles();

                    // Mark as success in results and CLEAR existing_id to remove the link
                    setUploadResults(prev => prev.map(r =>
                        r.name === fileObj.name ? { ...r, success: true, error: null, id: data.file_id, existing_id: null } : r
                    ));

                    setTimeout(() => {
                        setUploading(false);
                    }, 800);
                } catch (e) {
                    console.error("Replacement failed:", e);
                    alert("Overwriting failed: " + e.message);
                    setUploading(false);
                    setUploadShowResults(true);
                }
            };

            const handleLogout = async () => {
                try {
                    await api.logout();
                } catch (error) {
                    console.warn('Server logout failed; local session was cleared.', error);
                }
                clearAuthSession();
                localStorage.removeItem('vault_pinned_sessions');
                uploadAbortRef.current?.abort();
                uploadAbortRef.current = null;
                sessionSuspendedRef.current = false;
                setToken(null);
                setUser(null);
                setSessionExpired(false);
                setSessionExpiredDetail(null);
                setSessions([]);
                setCurrentSessionId(null);
                setPgChats([]);
                setPinnedSessions([]);
            };

            if (backendStatus === 'checking') return <CheckingPage />;
            if (backendStatus === 'low_memory') return <BackendErrorPage memory={backendMemory} onRetry={() => setBackendStatus('checking')} />;
            if (backendStatus === 'unreachable') return <ConnectionErrorPage onRetry={() => setBackendStatus('checking')} />;

            if (!token) {
                return <Login onLogin={(u) => {
                    sessionSuspendedRef.current = false;
                    setToken(localStorage.getItem('token'));
                    setUser(u);
                    setSessionExpiredDetail(null);
                }} initialUsername={user || ''} oauthExchangeError={oauthExchangeError} />;
            }

            return (
                <div className="flex h-screen bg-black text-white overflow-hidden relative">
                    {showUsernamePrompt && token && (
                        <UsernameSetupModal
                            currentUsername={user}
                            token={token}
                            onConfirm={(newUsername) => {
                                setUser(newUsername);
                                setShowUsernamePrompt(false);
                            }}
                        />
                    )}
                    <Sidebar
                        view={view}
                        setView={setView}
                        user={user}
                        userProfile={userProfile}
                        onLogout={handleLogout}
                        sessions={sessions}
                        currentSessionId={currentSessionId}
                        onSelectSession={(sid) => { setCurrentSessionId(sid); setCurrentPgChatId(null); }}
                        onNewChat={handleNewChat}
                        onDeleteSession={handleDeleteSession}
                        isCollapsed={sidebarCollapsed}
                        setIsCollapsed={setSidebarCollapsed}
                        isMobileOpen={isMobileOpen}
                        setIsMobileOpen={setIsMobileOpen}
                        storageUsed={totalSizeBytes}
                        storageQuota={storageQuotaBytes}
                        version={appVersion}
                        aiStats={aiStats}
                        pgChats={pgChats}
                        onDeletePgChat={handleDeletePgChat}
                        onPinChat={handlePinChat}
                        onOpenPgChat={handleOpenPgChat}
                        currentPgChatId={currentPgChatId}
                        pinnedSessions={pinnedSessions}
                        onTogglePin={togglePinSession}
                        currentTab={currentTab}
                        setCurrentTab={setCurrentTab}
                        onRefreshHome={refreshFiles}
                    />

                    <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
                        {/* Mobile Header */}
                        <div className="h-16 border-b border-white/5 bg-black flex items-center justify-between px-6 desktop-hide">
                            <button onClick={() => setIsMobileOpen(true)} className="p-2 -ml-2 text-zinc-400 hover:text-white">
                                <Icons.Menu size={24} />
                            </button>
                            <div className="flex items-center gap-2">
                                <img src="/svg/lavix.svg" alt="Lavix" className="h-7 w-auto filter brightness-110" />
                                <span className="text-sm font-black tracking-tighter gradient-text">LAVIX VAULT</span>
                            </div>
                            <div className="w-8"></div>
                        </div>

                        {/* Content row: scrollable main + details panel */}
                        <div className="flex flex-1 overflow-hidden">
                            {/* Scrollable main area — only this scrolls */}
                            <div className="flex-1 overflow-y-auto relative">
                                {view === 'files' && (
                                    <Dashboard
                                        files={files}
                                        searchTerm={searchTerm}
                                        setSearchTerm={setSearchTerm}
                                        setFiles={setFiles}
                                        refresh={refreshFiles}
                                        onPreview={setPreviewFile}
                                        currentTab={currentTab}
                                        setCurrentTab={setCurrentTab}
                                        onConfirmAction={onConfirmAction}
                                        totalSizeBytes={totalSizeBytes}
                                        aiStats={aiStats}
                                        onNavigateToChat={(ids) => { setChatPreTagIds(ids || []); setView('chat'); }}
                                        selectedItem={selectedItem}
                                        onSelectItem={setSelectedItem}
                                        filesLoading={filesLoading}
                                        onFoldersChange={setAppFolders}
                                        handleAiReadyClick={handleAiReadyClick}
                                        isIndexing={indexingActive}
                                        fileSearch={fileSearch}
                                        onUploadIntoFolder={(folderId) => { setUploadTargetFolder(folderId ?? null); setView('upload'); }}
                                    />
                                )}
                                {view === 'chat' && (
                                    <ChatView
                                        sessionId={currentSessionId}
                                        pgChatId={currentPgChatId}
                                        onPgChatCreated={handlePgChatCreated}
                                        onSessionUpdate={refreshSessions}
                                        onPreviewFile={setPreviewFile}
                                        onConfirmAction={onConfirmAction}
                                        aiStats={aiStats}
                                        files={files}
                                        folders={appFolders}
                                        fileSearch={fileSearch}
                                        toggleFileSearch={toggleFileSearch}
                                        isIndexing={indexingActive}
                                        handleAiReadyClick={handleAiReadyClick}
                                        onNavigateUpload={() => setView('upload')}
                                        preTagIds={chatPreTagIds}
                                        onGrantAccess={async (fileId) => {
                                            try {
                                                await api.grantAIAccess(fileId);
                                                await refreshFiles();
                                            } catch (e) {
                                                console.error('Grant access failed:', e);
                                            }
                                        }}
                                        refresh={refreshFiles}
                                        prefTimestamps={prefTimestamps}
                                        prefFontSize={prefFontSize}
                                        prefSendOnEnter={prefSendOnEnter}
                                        prefAutoScroll={prefAutoScroll}
                                    />
                                )}
                                {view === 'upload' && (
                                    <UploadView
                                        files={uploadFiles}
                                        uploading={uploading}
                                        progress={uploadProgress}
                                        uploadResults={uploadResults}
                                        showResults={uploadShowResults}
                                        currentFileIndex={uploadCurrentIndex}
                                        estimatedRemaining={estimatedRemaining}
                                        handleFiles={handleUploadFiles}
                                        handleFilePairs={handleUploadPairs}
                                        handleDismissResults={handleDismissUploadResults}
                                        onMinimize={() => { setMinimized(true); setView('files'); }}
                                        onPreview={setPreviewFile}
                                        onReplace={handleReplaceFile}
                                        folders={appFolders}
                                        uploadTargetFolder={uploadTargetFolder}
                                        setUploadTargetFolder={setUploadTargetFolder}
                                        onFoldersRefresh={async () => { const f = await api.getFolders().catch(()=>[]); setAppFolders(f); }}
                                    />
                                )}
                                {view === 'settings' && (
                                    <SettingsView
                                        userProfile={userProfile}
                                        onProfileUpdate={p => setUserProfile(p)}
                                        onUsernameChange={newUsername => setUser(newUsername)}
                                        prefFontSize={prefFontSize}
                                        setPrefFontSize={setPrefFontSize}
                                        prefTimestamps={prefTimestamps}
                                        setPrefTimestamps={setPrefTimestamps}
                                        prefSendOnEnter={prefSendOnEnter}
                                        setPrefSendOnEnter={setPrefSendOnEnter}
                                        prefAutoScroll={prefAutoScroll}
                                        setPrefAutoScroll={setPrefAutoScroll}
                                        onConfirmAction={onConfirmAction}
                                    />
                                )}
                            </div>

                        </div>
                    </div>

                    {/* Floating details panel — fixed overlay, does not push content */}
                    {selectedItem && view === 'files' && (
                        <FileDetails
                            file={selectedItem}
                            onClose={() => setSelectedItem(null)}
                            onPreview={setPreviewFile}
                            onDownload={api.downloadFile}
                            onGrantAccess={handleGrantAccessForPanel}
                            onCancelIndexing={handleCancelIndexingForPanel}
                            onRevokeAccess={handleRevokeAccessForPanel}
                            onSearch={setSearchTerm}
                            onChatAboutFile={handleChatAboutFile}
                        />
                    )}

                    <GlobalUploadWidget
                        uploading={uploading && view !== 'upload'}
                        progress={uploadProgress}
                        currentFileIndex={uploadCurrentIndex}
                        totalFiles={uploadFiles.length}
                        onMaximize={() => { setMinimized(false); setView('upload'); }}
                    />

                    {previewFile && <FileViewer file={previewFile} onClose={() => setPreviewFile(null)} onDetails={(f) => {
                        const fid = Number(f?.file_id ?? f?.id);
                        const full = Number.isInteger(fid) ? files.find(x => Number(x.id) === fid) : null;
                        setPreviewFile(null);
                        setView('files');
                        setSelectedItem(full || f);
                    }} />}
                    {confirmState && <ConfirmModal {...confirmState} onCancel={() => setConfirmState(null)} />}

                    {showDSConfirm && (
                        <div className="fixed inset-0 z-[250] flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in">
                            <div className="glass-card p-8 max-w-md w-full border-red-500/30 shadow-[0_0_50px_rgba(124,58,237,0.2)]">
                                <div className="flex items-center gap-4 mb-6" style={{ color: indexingActive ? '#f59e0b' : '#f87171' }}>
                                    <Icons.Zap size={32} />
                                    <h2 className="text-xl font-bold text-white">{indexingActive ? 'Indexing in Progress' : 'Enable File Indexing?'}</h2>
                                </div>

                                <p className="text-gray-300 mb-6 leading-relaxed">
                                    {indexingActive
                                        ? 'Lavix AI is currently indexing your files. Stopping cancels only queued and in-progress work; existing searchable revisions, embeddings, summaries, and tags stay intact.'
                                        : 'Lavix AI will read and remember all your files to provide accurate context-aware answers.'
                                    }
                                </p>

                                {(() => {
                                    const progress = getIndexProgress(aiStats);
                                    // Honest denominator: consented AND
                                    // parser-supported files only. The vault
                                    // may hold many unconsented files that
                                    // must not read as an indexing backlog.
                                    const total = progress.indexable;
                                    const ready = progress.searchable;
                                    const needed = Math.max(0, total - ready);
                                    // Cancelled-but-granted files sit inside
                                    // `needed` yet are retry work, not fresh
                                    // backlog: ACTIVATE re-queues them. Split
                                    // them out so the headline never reads as
                                    // stuck progress against a 100% badge.
                                    // NOTE: the list API serializes consent as
                                    // `ai_access_granted` (not the DB column
                                    // `user_granted_ai_access`) — filtering on
                                    // the wrong key silently empties (or fills)
                                    // these rows.
                                    //
                                    // One awaiting bucket: grantable-but-
                                    // ungranted files plus granted-cancelled
                                    // files without a revision. Each entry
                                    // carries its reason so cancelled work is
                                    // never double-counted and never hidden.
                                    const awaitingFiles = (files || []).flatMap(f => {
                                        if (f.is_deleted || isIndexSearchable(f)) return [];
                                        if (!f.ai_access_granted && f.parser_supported === true) {
                                            return [{ file: f, reason: 'needs grant' }];
                                        }
                                        if (f.ai_access_granted && f.ai_status === 'cancelled') {
                                            return [{ file: f, reason: 'cancelled — ACTIVATE retries' }];
                                        }
                                        return [];
                                    });
                                    const awaitingCount = awaitingFiles.length;
                                    const retryCount = awaitingFiles.filter(e => e.reason !== 'needs grant').length;
                                    const freshNeeded = Math.max(0, needed - retryCount);
                                    // Failed / password-locked without a
                                    // revision: terminal without user action.
                                    // Same filter feeds the static row and the
                                    // expandable details below — one array, no
                                    // drift. (Note: must use ai_access_granted;
                                    // the DB column name never reaches the client.)
                                    const attentionFiles = (files || []).filter(f =>
                                        !f.is_deleted && f.ai_access_granted &&
                                        (f.ai_status === 'password_required' || f.ai_status === 'failed')
                                    );

                                    let seconds = (needed * 5) || 5;
                                    let timeStr = seconds + " seconds";
                                    if (seconds > 60) timeStr = Math.ceil(seconds / 60) + " minutes";

                                    return (
                                        <div className="bg-white/5 rounded-xl p-4 mb-6 border border-white/10">
                                            <div className="flex justify-between items-center text-sm mb-2">
                                                <span className="text-zinc-400">{indexingActive ? 'Searchable' : 'Files to Index'}:</span>
                                                <span className="font-bold text-white">{indexingActive ? `${ready}/${total} files` : `${freshNeeded} files left`}</span>
                                            </div>
                                            {!indexingActive && awaitingCount > 0 && (
                                                <div className="mt-2 pt-2 border-t border-white/10">
                                                    <details>
                                                        <summary className="flex justify-between items-center text-sm cursor-pointer list-none">
                                                            <span className="text-zinc-400">Awaiting action:</span>
                                                            <span className="font-bold text-indigo-300">{awaitingCount} file{awaitingCount === 1 ? '' : 's'}</span>
                                                        </summary>
                                                        <div className="mt-2 space-y-1.5 max-h-40 overflow-y-auto">
                                                            {awaitingFiles.slice(0, 12).map(({ file: f, reason }) => (
                                                                <div key={f.id} className="flex items-center justify-between gap-2 text-xs">
                                                                    <span className="text-zinc-300 truncate" title={f.original_filename || f.filename}>{f.original_filename || f.filename}</span>
                                                                    <span className="text-zinc-500 whitespace-nowrap flex-shrink-0">{reason}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </details>
                                                </div>
                                            )}
                                            {indexingActive && (
                                                <div className="flex justify-between items-center text-xs mb-2">
                                                    <span className="text-zinc-500">Active / queued:</span>
                                                    <span className="font-semibold text-amber-300">{aiStats?.active || 0} / {aiStats?.queued || 0}</span>
                                                </div>
                                            )}
                                            {!indexingActive && (
                                                <div className="flex justify-between items-center text-sm mb-2">
                                                    <span className="text-zinc-400">Estimated Time:</span>
                                                    <span className="font-bold text-red-400">~{timeStr}</span>
                                                </div>
                                            )}
                                            {(() => {
                                                const mediaFiles = (files || []).filter(f => {
                                                    if (f.is_deleted) return false;
                                                    const mime = String(f.mime_type || '').toLowerCase();
                                                    return mime.startsWith('audio/') || mime.startsWith('video/');
                                                });
                                                if (!mediaFiles.length) return null;
                                                return (
                                                    <div className="mt-2 pt-2 border-t border-white/10">
                                                        <details>
                                                            <summary className="flex justify-between items-center text-xs cursor-pointer list-none">
                                                                <span className="text-zinc-500">Video/Audio (skipped):</span>
                                                                <span className="text-zinc-500">{mediaFiles.length} file{mediaFiles.length === 1 ? '' : 's'}</span>
                                                            </summary>
                                                            <div className="mt-2 space-y-1.5 max-h-40 overflow-y-auto">
                                                                {mediaFiles.slice(0, 12).map(f => (
                                                                    <div key={f.id} className="flex items-center justify-between gap-2 text-xs">
                                                                        <span className="text-zinc-300 truncate" title={f.original_filename || f.filename}>{f.original_filename || f.filename}</span>
                                                                        <span className="text-zinc-500 whitespace-nowrap flex-shrink-0">Audio/video isn't parsed</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </details>
                                                    </div>
                                                );
                                            })()}
                                            {attentionFiles.length > 0 && (
                                                <div className="flex justify-between items-center text-xs mt-2 pt-2 border-t border-white/10">
                                                    <span className="text-zinc-500">Failed / password-locked:</span>
                                                    <span className="font-semibold text-red-300">{attentionFiles.length} file{attentionFiles.length === 1 ? '' : 's'}</span>
                                                </div>
                                            )}
                                            {(() => {
                                                const unsupported = (files || []).filter(f => {
                                                    if (f.is_deleted || f.parser_supported !== false) return false;
                                                    const mime = String(f.mime_type || '').toLowerCase();
                                                    return !mime.startsWith('audio/') && !mime.startsWith('video/');
                                                });
                                                if (!unsupported.length) return null;
                                                return (
                                                    <div className="mt-2 pt-2 border-t border-white/10">
                                                        <details>
                                                            <summary className="flex justify-between items-center text-xs cursor-pointer list-none">
                                                                <span className="text-zinc-500">Unsupported format:</span>
                                                                <span className="text-zinc-500">{unsupported.length} file{unsupported.length === 1 ? '' : 's'}</span>
                                                            </summary>
                                                            <div className="mt-2 space-y-1.5 max-h-40 overflow-y-auto">
                                                                {unsupported.slice(0, 12).map(f => (
                                                                    <div key={f.id} className="flex items-center justify-between gap-2 text-xs">
                                                                        <span className="text-zinc-300 truncate" title={f.original_filename || f.filename}>{f.original_filename || f.filename}</span>
                                                                        <span className="text-zinc-500 whitespace-nowrap flex-shrink-0">No parser for this format</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </details>
                                                    </div>
                                                );
                                            })()}
                                            {(() => {
                                                const attention = attentionFiles;
                                                if (!attention.length) return null;
                                                return (
                                                    <div className="mt-2 pt-2 border-t border-white/10">
                                                        <details>
                                                            <summary className="flex justify-between items-center text-xs cursor-pointer list-none">
                                                                <span className="text-zinc-500">Needs attention:</span>
                                                                <span className="font-semibold text-red-300">{attention.length} file{attention.length === 1 ? '' : 's'}</span>
                                                            </summary>
                                                            <div className="mt-2 space-y-1.5 max-h-40 overflow-y-auto">
                                                                {attention.slice(0, 12).map(f => (
                                                                    <div key={f.id} className="flex items-center justify-between gap-2 text-xs">
                                                                        <span className="text-zinc-300 truncate" title={f.original_filename || f.filename}>{f.original_filename || f.filename}</span>
                                                                        <span className="text-zinc-500 whitespace-nowrap flex-shrink-0">
                                                                            {f.ai_status === 'password_required' ? 'Locked — needs file password' : 'Failed — retry from its row'}
                                                                        </span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </details>
                                                    </div>
                                                );
                                            })()}
                                        </div>
                                    );
                                })()}

                                <p className="text-xs text-zinc-500 mb-6 italic">
                                    Processing runs in the background. Cancelling never revokes AI access or deletes an existing index.
                                </p>

                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setShowDSConfirm(false)}
                                        className="flex-1 py-3 rounded-xl border border-white/10 text-gray-400 hover:text-white hover:bg-white/5 transition-all font-bold tracking-wide text-xs uppercase"
                                    >
                                        {indexingActive ? 'Close' : 'Reject'}
                                    </button>
                                    {indexingActive ? (
                                        <button
                                            onClick={cancelIndexing}
                                            className="flex-[2] py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white transition-all font-bold tracking-wide text-xs uppercase"
                                        >
                                            Cancel Indexing
                                        </button>
                                    ) : (
                                        <button
                                            onClick={confirmDeepSearch}
                                            className="flex-[2] py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white transition-all font-bold tracking-wide text-xs uppercase"
                                        >
                                            Activate
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {sessionExpired && (
                        <div className="fixed inset-0 z-[300] bg-black/80 backdrop-blur-md flex items-center justify-center p-4 animate-fade-in">
                            <div className="w-full max-w-md bg-[#0f0f10] border border-white/10 rounded-3xl overflow-hidden shadow-2xl relative">
                                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-orange-500 to-red-500 animate-pulse"></div>
                                <div className="p-8">
                                    <div className="flex flex-col items-center mb-6">
                                        <img src="/svg/lavix.svg" alt="Lavix Vault" className="w-16 h-16 object-contain mb-3" />
                                        <span className="text-lg font-black tracking-tight gradient-text">LAVIX VAULT</span>
                                    </div>
                                    <h2 className="text-2xl font-bold text-white text-center mb-2">
                                        {sessionExpiredDetail?.code === 'session_idle_timeout' ? 'Session Timed Out' : 'Session Suspended'}
                                    </h2>
                                    <p className="text-zinc-400 text-center text-sm mb-8">
                                        {sessionExpiredDetail?.code === 'session_idle_timeout'
                                            ? `Your session was locked after ${Math.round((sessionExpiredDetail.idle_timeout_seconds || 7200) / 60)} minutes without activity. Please re-authenticate to continue.`
                                            : 'Your security token has expired. Please re-authenticate to resume your session.'}
                                        {' '}Don't worry, your work is preserved.
                                    </p>

                                    <Login
                                        onLogin={(u) => {
                                            sessionSuspendedRef.current = false;
                                            setToken(localStorage.getItem('token'));
                                            setSessionExpired(false);
                                            setSessionExpiredDetail(null);
                                        }}
                                        onLogout={handleLogout}
                                        embedded={true}
                                        initialUsername={user}
                                        lockUsername={true}
                                    />
                                </div>
                            </div>


                        </div>
                    )}
                </div>
            );
        }



export { ErrorBoundary }
export default App

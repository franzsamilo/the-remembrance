// Custom SVG Icons for The Lexicon Vault
// Archival-themed icons with unique styling

export const ArchiveBookIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Book with archival seal */}
    <rect x="5" y="3" width="14" height="18" rx="1" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <rect x="6" y="4" width="12" height="16" rx="0.5" fill="currentColor" opacity="0.1"/>
    <line x1="8" y1="7" x2="16" y2="7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <line x1="8" y1="10" x2="16" y2="10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <line x1="8" y1="13" x2="14" y2="13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <circle cx="16" cy="17" r="2.5" fill="currentColor" opacity="0.2"/>
    <path d="M 16 15.5 L 16 18.5 M 14.5 17 L 17.5 17" stroke="currentColor" strokeWidth="1" strokeLinecap="round"/>
  </svg>
);

export const KnowledgeGraphIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Network nodes with connections */}
    <circle cx="12" cy="12" r="2.5" fill="currentColor"/>
    <circle cx="6" cy="6" r="2" fill="currentColor" opacity="0.7"/>
    <circle cx="18" cy="6" r="2" fill="currentColor" opacity="0.7"/>
    <circle cx="6" cy="18" r="2" fill="currentColor" opacity="0.7"/>
    <circle cx="18" cy="18" r="2" fill="currentColor" opacity="0.7"/>
    <line x1="9" y1="10.5" x2="7.5" y2="7.5" stroke="currentColor" strokeWidth="1.5" opacity="0.4"/>
    <line x1="15" y1="10.5" x2="16.5" y2="7.5" stroke="currentColor" strokeWidth="1.5" opacity="0.4"/>
    <line x1="9" y1="13.5" x2="7.5" y2="16.5" stroke="currentColor" strokeWidth="1.5" opacity="0.4"/>
    <line x1="15" y1="13.5" x2="16.5" y2="16.5" stroke="currentColor" strokeWidth="1.5" opacity="0.4"/>
    <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1" opacity="0.2" fill="none"/>
  </svg>
);

export const WaxSealIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Wax seal with checkmark */}
    <circle cx="12" cy="12" r="9" fill="currentColor" opacity="0.15"/>
    <circle cx="12" cy="12" r="7" stroke="currentColor" strokeWidth="1.5"/>
    <circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
    <path d="M 9 12 L 11 14 L 15 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
  </svg>
);

export const ScrollIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Ancient scroll */}
    <path d="M 6 4 Q 4 4 4 6 L 4 18 Q 4 20 6 20" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <path d="M 18 4 Q 20 4 20 6 L 20 18 Q 20 20 18 20" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <rect x="6" y="4" width="12" height="16" fill="currentColor" opacity="0.1"/>
    <line x1="8" y1="8" x2="16" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.6"/>
    <line x1="8" y1="11" x2="16" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.6"/>
    <line x1="8" y1="14" x2="14" y2="14" stroke="currentColor" strokeWidth="1" opacity="0.6"/>
  </svg>
);

export const ArchiveBoxIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Archive box with label */}
    <rect x="4" y="8" width="16" height="12" rx="1" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <rect x="4" y="4" width="16" height="4" rx="1" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="9" y="12" width="6" height="2" rx="0.5" fill="currentColor" opacity="0.4"/>
    <line x1="7" y1="15" x2="17" y2="15" stroke="currentColor" strokeWidth="0.5" opacity="0.3"/>
    <line x1="7" y1="17" x2="17" y2="17" stroke="currentColor" strokeWidth="0.5" opacity="0.3"/>
  </svg>
);

export const QuillPenIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Quill pen for writing */}
    <path d="M 20 4 Q 18 6 16 8 L 8 16 L 4 20 L 6 18 L 8 16 L 16 8 Q 18 6 20 4 Z" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
    <path d="M 16 8 L 8 16" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
    <circle cx="6" cy="18" r="1.5" fill="currentColor"/>
    <path d="M 18 6 Q 19 5 20 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

export const CatalogIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Card catalog */}
    <rect x="5" y="4" width="14" height="16" rx="1" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <rect x="6" y="5" width="12" height="14" rx="0.5" fill="currentColor" opacity="0.05"/>
    <rect x="8" y="10" width="8" height="1.5" rx="0.5" fill="currentColor" opacity="0.3"/>
    <line x1="8" y1="7" x2="16" y2="7" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
    <line x1="8" y1="14" x2="16" y2="14" stroke="currentColor" strokeWidth="0.5" opacity="0.3"/>
    <line x1="8" y1="16" x2="14" y2="16" stroke="currentColor" strokeWidth="0.5" opacity="0.3"/>
    <circle cx="17" cy="10.75" r="0.75" fill="currentColor"/>
  </svg>
);

export const ManuscriptIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Manuscript pages */}
    <rect x="6" y="3" width="12" height="18" rx="1" fill="currentColor" opacity="0.1" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="7" y="4" width="10" height="16" rx="0.5" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.3"/>
    <line x1="9" y1="7" x2="15" y2="7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <line x1="9" y1="10" x2="15" y2="10" stroke="currentColor" strokeWidth="1" opacity="0.6"/>
    <line x1="9" y1="12.5" x2="15" y2="12.5" stroke="currentColor" strokeWidth="1" opacity="0.6"/>
    <line x1="9" y1="15" x2="13" y2="15" stroke="currentColor" strokeWidth="1" opacity="0.6"/>
    <path d="M 9 7 L 8 6" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.4"/>
  </svg>
);

export const VaultIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Vault door */}
    <rect x="4" y="4" width="16" height="16" rx="1" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <rect x="5" y="5" width="14" height="14" rx="0.5" fill="currentColor" opacity="0.05"/>
    <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <circle cx="12" cy="12" r="2" fill="currentColor" opacity="0.3"/>
    <line x1="12" y1="12" x2="14.5" y2="9.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <rect x="18" y="11" width="2" height="2" rx="0.5" fill="currentColor"/>
  </svg>
);

export const InkwellIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Inkwell with quill */}
    <path d="M 8 14 L 8 18 Q 8 20 10 20 L 14 20 Q 16 20 16 18 L 16 14 Z" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="7" y="12" width="10" height="2" rx="0.5" fill="currentColor" opacity="0.3"/>
    <path d="M 18 4 L 14 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M 18 4 Q 19 3 20 2" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.6"/>
  </svg>
);

export const SettingsGearIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Ornate gear */}
    <circle cx="12" cy="12" r="3" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M 12 4 L 12 6 M 12 18 L 12 20 M 4 12 L 6 12 M 18 12 L 20 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M 7.5 7.5 L 8.8 8.8 M 15.2 15.2 L 16.5 16.5 M 7.5 16.5 L 8.8 15.2 M 15.2 8.8 L 16.5 7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    <circle cx="12" cy="12" r="7" stroke="currentColor" strokeWidth="0.5" opacity="0.2" fill="none"/>
  </svg>
);

export const DatabaseIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    {/* Stacked archive */}
    <ellipse cx="12" cy="6" rx="7" ry="2.5" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M 5 6 L 5 18 Q 5 20 12 20 Q 19 20 19 18 L 19 6" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <ellipse cx="12" cy="12" rx="7" ry="2" stroke="currentColor" strokeWidth="1" opacity="0.4" fill="none"/>
    <ellipse cx="12" cy="18" rx="7" ry="2" stroke="currentColor" strokeWidth="1" opacity="0.4" fill="none"/>
  </svg>
);

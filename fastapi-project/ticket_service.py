"""
Ticket service with ChromaDB integration for semantic search
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not installed. Semantic search will be disabled.")

from models import Ticket
from config import settings


class TicketService:
    """
    Service for managing tickets with optional ChromaDB integration
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger("TicketService")
        self.chroma_client = None
        self.collection = None
        
        # Initialize ChromaDB if enabled and available
        if settings.use_chromadb and CHROMADB_AVAILABLE:
            try:
                self._init_chromadb()
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB: {e}")
                self.logger.warning("Falling back to PostgreSQL-only mode")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.chroma_client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.chromadb_collection,
                metadata={"description": "Test automation tickets"}
            )
            
            self.logger.info(f"✅ ChromaDB initialized: {settings.chromadb_collection}")
            
        except Exception as e:
            self.logger.error(f"ChromaDB initialization failed: {e}")
            raise
    
    def create_ticket_from_file(
        self,
        ticket_id: str,
        project_id: int,
        file_path: str,
        title: Optional[str] = None,
        module: Optional[str] = None
    ) -> Ticket:
        """
        Create ticket from file and store in PostgreSQL + ChromaDB
        """
        # Check if ticket already exists
        existing = self.db.query(Ticket).filter(
            Ticket.ticket_id == ticket_id
        ).first()
        
        if existing:
            self.logger.warning(f"Ticket {ticket_id} already exists, updating...")
            ticket = existing
        else:
            ticket = Ticket(
                ticket_id=ticket_id,
                project_id=project_id,
                created_at=datetime.now()
            )
            self.db.add(ticket)
        
        # Update ticket details
        ticket.file_path = file_path
        ticket.title = title or f"Test: {ticket_id}"
        ticket.module = module or "Unknown"
        
        # Parse file content
        try:
            content = self._read_ticket_file(file_path)
            ticket.description = content.get('description', '')
            ticket.acceptance_criteria = content.get('acceptance_criteria', '')
            
            # Extract and store test steps as JSON
            steps = content.get('steps', [])
            ticket.test_steps = steps  # Store as JSONB
            
        except Exception as e:
            self.logger.error(f"Failed to parse ticket file: {e}")
        
        # Save to PostgreSQL
        self.db.commit()
        self.db.refresh(ticket)
        
        self.logger.info(f"✅ Ticket {ticket_id} saved to PostgreSQL")
        
        # Index in ChromaDB for semantic search
        if self.collection:
            try:
                self._index_ticket_in_chromadb(ticket)
            except Exception as e:
                self.logger.error(f"Failed to index in ChromaDB: {e}")
        
        return ticket
    
    def _read_ticket_file(self, file_path: str) -> Dict:
        """
        Parse ticket file and extract structured information
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Ticket file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract sections
        description = self._extract_section(content, ['Description', 'Summary'])
        acceptance_criteria = self._extract_section(content, ['Acceptance Criteria', 'AC'])
        steps = self._extract_steps(content)
        
        return {
            'description': description,
            'acceptance_criteria': acceptance_criteria,
            'steps': steps,
            'full_content': content
        }
    
    def _extract_section(self, content: str, headers: List[str]) -> str:
        """Extract content between section headers"""
        for header in headers:
            pattern = rf"{header}[:\s]*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_steps(self, content: str) -> List[Dict]:
        """
        Extract test steps from ticket content
        Supports formats:
        - Step 1: ...
        - 1. ...
        - Step 1. ...
        """
        steps = []
        
        # Pattern to match step lines
        patterns = [
            r'(?:Step\s+)?(\d+)[.:]\s*(.+?)(?=(?:Step\s+)?\d+[.:]|\Z)',
            r'^\s*(\d+)\.\s+(.+?)$'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            for match in matches:
                step_num = int(match.group(1))
                step_text = match.group(2).strip()
                
                # Clean up step text (remove extra whitespace)
                step_text = ' '.join(step_text.split())
                
                if step_text:
                    steps.append({
                        'num': step_num,
                        'text': step_text
                    })
        
        # Remove duplicates and sort
        unique_steps = {}
        for step in steps:
            if step['num'] not in unique_steps:
                unique_steps[step['num']] = step
        
        return sorted(unique_steps.values(), key=lambda x: x['num'])
    
    def _index_ticket_in_chromadb(self, ticket: Ticket):
        """
        Index ticket in ChromaDB for semantic search
        """
        if not self.collection:
            return
        
        # Prepare document for indexing
        doc_text = f"""
        Ticket: {ticket.ticket_id}
        Title: {ticket.title or ''}
        Module: {ticket.module or ''}
        Description: {ticket.description or ''}
        Acceptance Criteria: {ticket.acceptance_criteria or ''}
        Steps: {' '.join([s['text'] for s in (ticket.test_steps or [])])}
        """.strip()
        
        # Add to ChromaDB
        self.collection.upsert(
            ids=[ticket.ticket_id],
            documents=[doc_text],
            metadatas=[{
                "ticket_id": ticket.ticket_id,
                "project_id": ticket.project_id,
                "title": ticket.title or "",
                "module": ticket.module or "",
                "created_at": ticket.created_at.isoformat()
            }]
        )
        
        self.logger.info(f"✅ Ticket {ticket.ticket_id} indexed in ChromaDB")
    
    def get_ticket_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """
        Get ticket by ID from PostgreSQL
        """
        return self.db.query(Ticket).filter(
            Ticket.ticket_id == ticket_id
        ).first()
    
    def search_tickets(
        self,
        query: str,
        limit: int = 10,
        use_semantic: bool = True
    ) -> List[Ticket]:
        """
        Search tickets using ChromaDB (semantic) or PostgreSQL (exact match)
        """
        if use_semantic and self.collection:
            return self._semantic_search(query, limit)
        else:
            return self._postgres_search(query, limit)
    
    def _semantic_search(self, query: str, limit: int) -> List[Ticket]:
        """
        Semantic search using ChromaDB
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            ticket_ids = results['ids'][0] if results['ids'] else []
            
            # Fetch full tickets from PostgreSQL
            tickets = self.db.query(Ticket).filter(
                Ticket.ticket_id.in_(ticket_ids)
            ).all()
            
            # Sort by ChromaDB relevance order
            ticket_map = {t.ticket_id: t for t in tickets}
            sorted_tickets = [ticket_map[tid] for tid in ticket_ids if tid in ticket_map]
            
            self.logger.info(f"Semantic search found {len(sorted_tickets)} tickets")
            return sorted_tickets
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return self._postgres_search(query, limit)
    
    def _postgres_search(self, query: str, limit: int) -> List[Ticket]:
        """
        Fallback search using PostgreSQL LIKE
        """
        tickets = self.db.query(Ticket).filter(
            (Ticket.ticket_id.ilike(f"%{query}%")) |
            (Ticket.title.ilike(f"%{query}%")) |
            (Ticket.description.ilike(f"%{query}%"))
        ).limit(limit).all()
        
        self.logger.info(f"PostgreSQL search found {len(tickets)} tickets")
        return tickets
    
    def list_all_tickets(self, project_id: Optional[int] = None) -> List[Ticket]:
        """
        List all tickets, optionally filtered by project
        """
        query = self.db.query(Ticket)
        
        if project_id:
            query = query.filter(Ticket.project_id == project_id)
        
        return query.order_by(Ticket.created_at.desc()).all()
    
    def delete_ticket(self, ticket_id: str) -> bool:
        """
        Delete ticket from PostgreSQL and ChromaDB
        """
        ticket = self.get_ticket_by_id(ticket_id)
        
        if not ticket:
            return False
        
        # Delete from PostgreSQL
        self.db.delete(ticket)
        self.db.commit()
        
        # Delete from ChromaDB
        if self.collection:
            try:
                self.collection.delete(ids=[ticket_id])
                self.logger.info(f"Deleted {ticket_id} from ChromaDB")
            except Exception as e:
                self.logger.error(f"Failed to delete from ChromaDB: {e}")
        
        self.logger.info(f"✅ Ticket {ticket_id} deleted")
        return True
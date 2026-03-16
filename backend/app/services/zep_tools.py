"""
Zep Retrieval Tool Service
Encapsulates graph search, node reading, edge queries and other tools for use by Report agents

Core Retrieval Tools (Optimized):
1. InsightForge (Deep Insight Retrieval) - Most powerful hybrid retrieval, automatically generates sub-questions and retrieves from multiple dimensions
2. PanoramaSearch (Breadth Search) - Get the full picture, including expired content
3. QuickSearch (Simple Search) - Fast retrieval
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class Searchresults:
    """Search results"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Convert to text format for LLM understanding"""
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} related information items"]
        
        if self.facts:
            text_parts.append("\n### Related facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node Information"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown type")
        return f"Entity: {self.name} (Type: {entity_type})\nsummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge Information"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Temporal information
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "To date"
            base_text += f"\nValidity period: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """Whether expired"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """Whether invalid"""
        return self.invalid_at is not None


@dataclass
class InsightForgeresults:
    """
    Deep insight retrieval result (InsightForge)
    Contains multiple sub-questions retrieval result, and comprehensive analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # Multi-dimensional retrieval results
    semantic_facts: List[str] = field(default_factory=list)  # Semantic search results
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entity insights
    relationship_chains: List[str] = field(default_factory=list)  # Relationship chains
    
    # Statistical information
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding"""
        text_parts = [
            f"## Deep analysis of future predictions",
            f"Analysis question: {self.query}",
            f"PredictionScenario: {self.simulation_requirement}",
            f"\n### Prediction data statistics",
            f"- Related prediction facts: {self.total_facts}Items",
            f"- Involved entities: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}Items"
        ]
        
        # sub-questions
        if self.sub_queries:
            text_parts.append(f"\n### Sub-questions analyzed")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # Semantic search results
        if self.semantic_facts:
            text_parts.append(f"\n### [Key facts](Please cite these original texts in the report)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Entity insights
        if self.entity_insights:
            text_parts.append(f"\n### [Core entities]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  summary: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))}Items")
        
        # Relationship chains
        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class Panoramaresults:
    """
    Breadth search result (Panorama)
    Contains all related information, including expired content
    """
    query: str
    
    # All nodes
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges(including expired)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Currently valid facts
    active_facts: List[str] = field(default_factory=list)
    # Expired/Invalid facts(Historical records)
    historical_facts: List[str] = field(default_factory=list)
    
    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """Convert to text format(complete version, no truncation)"""
        text_parts = [
            f"## Breadth search result (future panoramic view)",
            f"Query: {self.query}",
            f"\n### Statistical information",
            f"- Total number of nodes: {self.total_nodes}",
            f"- Total number of edges: {self.total_edges}",
            f"- Current valid facts: {self.active_count}Items",
            f"- History/Expired facts: {self.historical_count}Items"
        ]
        
        # Currently valid facts (complete output, no truncation)
        if self.active_facts:
            text_parts.append(f"\n### [Current valid facts](Original simulation results)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # History/Expired facts (complete output, no truncation)
        if self.historical_facts:
            text_parts.append(f"\n### [History/Expired facts] (evolution process records)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Key entities (complete output, no truncation)
        if self.all_nodes:
            text_parts.append(f"\n### [Involved entities]")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class agentsInterview:
    """Individual agent interview results"""
    agent_name: str
    agent_role: str  # Role type (such as: students, teachers, media, etc)
    agent_bio: str  # Brief introduction
    question: str  # Interview questions
    response: str  # Interview answers
    key_quotes: List[str] = field(default_factory=list)  # Key quotes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Display complete agent biography, no truncation
        text += f"_Brief introduction: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key quotes:**\n"
            for quote in self.key_quotes:
                # Clean up various quotation marks
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Remove starting punctuation (ASCII and fullwidth Chinese)
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter quotes that are just question headers (e.g. "Question 1", "问题1")
                skip = False
                for d in '123456789':
                    if f'Question {d}' in clean_quote or f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncate too long content (prefer sentence boundary)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)  # Chinese period
                    if dot_pos < 0:
                        dot_pos = clean_quote.find('.', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class Interviewresults:
    """
    Interview results (Interview)
    Contain interview answers from multiple simulation agents
    """
    interview_topic: str  # Interview topic
    interview_questions: List[str]  # Interview questionsList
    
    # Selected agent for interview
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Interview answers from each agent
    interviews: List[agentsInterview] = field(default_factory=list)
    
    # Reason for selecting agents
    selection_reasoning: str = ""
    # Integrated interview summary
    summary: str = ""
    
    # Statistics
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding and report citation"""
        text_parts = [
            "## Deep interview report",
            f"**Interview topic:** {self.interview_topic}",
            f"**Number of interviews:** {self.interviewed_count} / {self.total_agents} simulation agents",
            "\n### Reason for interview subject selection",
            self.selection_reasoning or "(Automatic selection)",
            "\n---",
            "\n### Interview records",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview summary and core viewpoints")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep retrieval tools service
    
    [Core retrieval tools - after optimization]
    1. insight_forge - Deep insight retrieval (most powerful, auto-generate sub-questions, multi-dimensional retrieval)
    2. panorama_search - Breadth search(Get full picture, including expired content)
    3. quick_search - Simple search(Quick retrieval)
    4. interview_agents - Deep interview(Interview simulation agents, Get multi-perspective views)
    
    [Basic tools]
    - search_graph - Graph semantic search
    - get_all_nodes - Get all nodes in the graph
    - get_all_edges - Get all graph edges (contains temporal information)
    - get_node_detail - Get detailed information for a specific node
    - get_node_edges - Get node related edges
    - get_entities_by_type - Get entity by type
    - get_entity_summary - Get entity relationship summary
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY not configured")
        
        self.client = Zep(api_key=self.api_key)
        # LLM client for InsightForge generate sub-questions
        self._llm_client = llm_client
        logger.info("ZepToolsService initialization complete")
    
    @property
    def llm(self) -> LLMClient:
        """Delay initialize LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """API call with retry mechanism"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} attempt {attempt + 1} failed: {str(e)[:100]}, "
                        f"{delay:.1f}seconds later, retry..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} still failed after {max_retries} attempts: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> Searchresults:
        """
        Graph semantic search
        
        Use hybrid search (semantic+BM25)Search related information in graph.
        If Zep Cloud search API is unavailable, downgrade to local keyword matching.
        
        Args:
            graph_id: GraphID (Standalone Graph)
            query: Search query
            limit: Number of results to return
            scope: Search range, "edges" or "nodes"
            
        Returns:
            Searchresults: Searchresults
        """
        logger.info(f"GraphSearch: graph_id={graph_id}, query={query[:50]}...")
        
        # Try to use Zep Cloud search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"GraphSearch(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Parse edge search result
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # Parse node search result
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # Node summary also counts as fact
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"Searchcomplete: Found {len(facts)} related facts")
            
            return Searchresults(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep search API failed, Downgrade to local search: {str(e)}")
            # Downgrade: use local keyword matching search
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> Searchresults:
        """
        Local keyword matching search (as Search API downgrade plan)
        
        Get all edges/Node, then perform local keyword matching
        
        Args:
            graph_id: GraphID
            query: Search query
            limit: Number of results to return
            scope: Search range
            
        Returns:
            Searchresults: Searchresults
        """
        logger.info(f"Use local search: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Extract query keywords(Simple tokenization)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace(', ', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """Calculate text and query matching score"""
            if not text:
                return 0
            text_lower = text.lower()
            # Exact match query
            if query_lower in text_lower:
                return 100
            # Keyword matching
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Get all edges and match
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # Sort by score
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Get all nodes and match
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"Local search complete: Found {len(facts)} related facts")
            
        except Exception as e:
            logger.error(f"Local search failed: {str(e)}")
        
        return Searchresults(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Get all nodes of graph(Paged retrieval)

        Args:
            graph_id: GraphID

        Returns:
            List of NodeInfo objects
        """
        logger.info(f"Get all nodes of graph {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"Got {len(result)} nodes")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Get all edges of graph(Paged retrieval, Contains time information)

        Args:
            graph_id: GraphID
            include_temporal: Whether to contain time information(default True)

        Returns:
            List of EdgeInfo objects (containing created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(f"Get all edges of graph {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # Add temporal information if requested
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"Got {len(result)} edges")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Get detailed information of individual node
        
        Args:
            node_uuid: NodeUUID
            
        Returns:
            NodeInfo object or None if not found
        """
        logger.info(f"Get node details: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"Get node details(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"Get node detailsfailed: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Get all edges related to node
        
        By getting all graph edges, then filter out edges related to the specified node
        
        Args:
            graph_id: GraphID
            node_uuid: NodeUUID
            
        Returns:
            EdgeList
        """
        logger.info(f"Get node {node_uuid[:8]}... related edges")
        
        try:
            # Get all graph edges, then filter
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Check if edge is related to specified node(as source or target)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"Found {len(result)} edges related to node")
            return result
            
        except Exception as e:
            logger.warning(f"Get node edges failed: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Get entity by type
        
        Args:
            graph_id: GraphID
            entity_type: Entity type(such as Student, PublicFigure etc)
            
        Returns:
            Entity list that matches type
        """
        logger.info(f"Get type {entity_type} entity...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Check if labels contain specified type
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"Found {len(filtered)} entities of type {entity_type}")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get relationship summary of specified entity
        
        Search all information related to the entity and generate summary
        
        Args:
            graph_id: GraphID
            entity_name: Entity name
            
        Returns:
            Entity summary information dictionary
        """
        logger.info(f"Get entity {entity_name} relationship summary...")
        
        # First search for information related to the entity
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # Try to find the entity in all nodes
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # pass graph_id parameter
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get statistical information of graph
        
        Args:
            graph_id: GraphID
            
        Returns:
            Statistical information
        """
        logger.info(f"Get graph {graph_id} statistical information...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # Count entity type distribution
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # Statistics relationship type distribution
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get context information related to simulation
        
        Comprehensively search all information related to simulation requirements
        
        Args:
            graph_id: GraphID
            simulation_requirement: Simulation requirement description
            limit: Quantity limit for each type of information
            
        Returns:
            Simulation context information dictionary
        """
        logger.info(f"Get simulation context: {simulation_requirement[:50]}...")
        
        # Search information related to simulation requirements
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Get graph statistics
        stats = self.get_graph_statistics(graph_id)

        # Get all entity nodes
        all_nodes = self.get_all_nodes(graph_id)
        
        # Filter entities with actual types(not pure entity nodes)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Limit quantity
            "total_entities": len(entities)
        }
    
    # ========== Core retrieval tools(after optimization) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeresults:
        """
        [InsightForge - Deep insight retrieval]
        
        Most powerful hybrid retrieval function, automatically decompose problems and multi-dimensional retrieval:
        1. Use LLM to convert problem decomposition to multiple sub-questions
        2. Perform semantic search for each sub-question
        3. Extract related entities and get their detailed information
        4. Track relationship chains
        5. Integrate all results and generate deep insights
        
        Args:
            graph_id: GraphID
            query: User question
            simulation_requirement: Simulation requirement description
            report_context: Report context(Optional, for more precise sub-question generation)
            max_sub_queries: Maximum sub-question quantity
            
        Returns:
            InsightForgeresults: Deep insight retrieval result
        """
        logger.info(f"InsightForge Deep insight retrieval: {query[:50]}...")
        
        result = InsightForgeresults(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Use LLM to generate sub-questions
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Generate {len(sub_queries)} sub-questions")
        
        # Step 2: Perform semantic search for each sub-question
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # Also search the original question
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Extract related entity UUIDs from edges, only get information for these entities(do not get all nodes)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Get details of all related entities (do not limit quantity, complete output)
        entity_insights = []
        node_map = {}  # for subsequent relationship chain construction
        
        for uuid in list(entity_uuids):  # Process all entities, no truncation
            if not uuid:
                continue
            try:
                # Separately get eachrelatedNodeinformation
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                    
                    # Get all facts related to the entity(no truncation)
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # complete output, no truncation
                    })
            except Exception as e:
                logger.debug(f"GetNode {uuid} failed: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Build all relationship chains(do not limit quantity)
        relationship_chains = []
        for edge_data in all_edges:  # Process all edges, no truncation
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForge complete: {result.total_facts} facts, {result.total_entities} entities, {result.total_relationships} relationships")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Use LLM to generate sub-questions
        
        Convert complex problem decomposition to multiple independently retrievable sub-questions
        """
        system_prompt = """You are a professional question analysis expert. Your task is to convert a complex problem into multiple sub-questions that can be independently observed in the simulation world.

Require:
1. Each sub-question should be specific enough, can find related agent behaviors or events in the simulation world
2. Sub-questions should cover different dimensions of the original question(such as: who, what, Towhat, how, when, where)
3. Sub-questions should be related to simulation scenario
4. Return JSON format:{"sub_queries": ["sub-questions1", "sub-questions2", ...]}"""

        user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please convert the following question into {max_queries} sub-questions:
{query}

Return a JSON format sub-questions list."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # Ensure it is a string list
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"Generate sub-questions failed: {str(e)}, Use default sub-questions")
            # Downgrade: return variants based on original question
            return [
                query,
                f"{query} main participants",
                f"{query} causes and effects",
                f"{query} development process"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> Panoramaresults:
        """
        [PanoramaSearch - Breadth search]
        
        Get a full-picture view of all related content, including historical and expired information:
        1. Get all related nodes
        2. Get all edges(Including expired/invalid)
        3. Classify and organize current valid and historical information
        
        This tool is suitable for scenarios that need to understand the full picture of events, scenarios for tracking evolution process.
        
        Args:
            graph_id: GraphID
            query: Search query(for relevance ranking)
            include_expired: Whether to include expired/historical content (default True)
            limit: Number of results to return limit
            
        Returns:
            Panoramaresults: Breadth search result
        """
        logger.info(f"PanoramaSearch Breadth search: {query[:50]}...")
        
        result = Panoramaresults(query=query)
        
        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Get all edges(Contains time information)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # Classify facts
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # Add entity names to facts
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # Determine if expired/invalid
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # History/Expired facts, Add time marker
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Current valid facts
                active_facts.append(edge.fact)
        
        # Sort by relevance based on query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace(', ', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Limit quantity
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"Panorama search complete: {result.active_count}items active, {result.historical_count}items history")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> Searchresults:
        """
        [QuickSearch - Simple search]
        
        Fast, lightweight retrieval tool:
        1. Directly call Zep semantic search
        2. Return most relevant results
        3. Suitable for simple and direct retrieval requirements
        
        Args:
            graph_id: GraphID
            query: Search query
            limit: Number of results to return
            
        Returns:
            Searchresults: Searchresults
        """
        logger.info(f"QuickSearch Simple search: {query[:50]}...")
        
        # Directly call existing search_graph method
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> Interviewresults:
        """
        [InterviewAgents - Deep interview]

        Call the real OASIS Interview API to interview simulation agents:
        1. Automatically read persona files to understand all simulated agents
        2. Use LLM to analyze the interview requirement and select the most relevant agents
        3. Use LLM to generate interview questions
        4. Call /api/simulation/interview/batch interface for the real interview (dual platform simultaneous)
        5. Integrate all interview results and generate the interview report

        [Important] This function requires the simulation environment to be running (OASIS environment not closed)
        
        [Use Cases]
        - Need to understand event perspectives from different role perspectives
        - Need to collect opinions and viewpoints from multiple parties
        - Need to get real answers from simulation agents (not LLM simulation)

        Args:
            simulation_id: Simulation ID (used to locate persona files and call the Interview API)
            interview_requirement: Interview requirement description (e.g. "understand students' views on events")
            simulation_requirement: Simulation requirement background (optional)
            max_agents: Maximum number of agents to interview
            custom_questions: Custom interview questions (optional, auto-generated if not provided)
            
        Returns:
            Interviewresults: Interview results
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(f"Interview agents deep interview(API): {interview_requirement[:50]}...")
        
        result = Interviewresults(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Read persona files
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(f"Persona file not found for simulation {simulation_id}")
            result.summary = "No agent persona files found for interview"
            return result
        
        result.total_agents = len(profiles)
        logger.info(f"Loaded {len(profiles)} agent personas")
        
        # Step 2: Use LLM to select agents to interview (returns agent ID list)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"Selected {len(selected_agents)} agents for interview: {selected_indices}")
        
        # Step 3: Generate interview questions (if not already provided via custom_questions)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"Generated {len(result.interview_questions)} interview questions")
        
        # Merge all interview questions into a single combined prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # Add optimization prefix to constrain agent response format
        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Please draw on your persona, all past memories and actions, "
            "and directly answer the following questions in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language, Do not call any tools\n"
            "2. Do not return JSON format ortool call format\n"
            "3. Do not use markdown(such as#, ##, ###)\n"
            "4. Answer questions one by one by number. Each answer starts with 'Question X:' as a header (X is the question number)\n"
            "5. Separate answers between questions with blank lines\n"
            "6. Answers should have substantive content, Answer at least 2-3 sentences per question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: Call the real Interview API (no platform specified = dual platform simultaneous interview)
        try:
            # Build batch interview list (platform=None means both platforms are interviewed simultaneously)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # Use the optimized prompt
                    # No platform specified: API will interview on both twitter and reddit
                })

            logger.info(f"Calling batch interview API (dual platform): {len(interviews_request)} agents")

            # Call SimulationRunner batch interview (platform=None = dual platform interview)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # No platform specified = interview on both platforms
                timeout=180.0   # Longer timeout for dual-platform interviews
            )
            
            logger.info(f"Interview API: {api_result.get('interviews_count', 0)} results, success={api_result.get('success')}")
            
            # Check if API call succeeded
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning(f"Interview API returned failure: {error_msg}")
                result.summary = f"Interview API call failed: {error_msg}. Please check the OASIS simulation environment status."
                return result

            # Step 5: Parse API results and build agent interview objects
            # Dual-platform result format: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"agents_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")
                
                # Get this agent's interview results from both platforms
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Clean up possible tool call JSON wrapper
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Label responses by platform
                twitter_text = twitter_response if twitter_response else "(No response from Twitter platform)"
                reddit_text = reddit_response if reddit_response else "(No response from Reddit platform)"
                response_text = f"[Twitter platform answer]\n{twitter_text}\n\n[Reddit platform answer]\n{reddit_text}"

                # Extract key quotes (from both platform responses)
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Clean response text: remove markers, special chars, Markdown formatting
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'question \d+[::]\s*', '', clean_text)
                clean_text = re.sub(r'\[[^\]]+\]', '', clean_text)

                # Strategy 1 (primary): Extract complete meaningful sentences
                sentences = re.split(r'[.!?！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W,;:]+', s.strip())
                    and not s.strip().startswith(('{', 'question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "." for s in meaningful[:3]]

                # Strategy 2 (fallback): Look for quoted text using common quote characters
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[,;:]', q)][:3]
                
                interview = agentsInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # expand bio lengthlimit
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Simulation environment is not running
            logger.warning(f"Interview API call failed (environment not running?): {e}")
            result.summary = f"Interview failed: {str(e)}. The simulation environment may not be running. Please ensure the OASIS environment is active."
            return result
        except Exception as e:
            logger.error(f"Interview API call exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Interview process error: {str(e)}"
            return result
        
        # Step 6: Generate interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(f"Interview agents complete: interviewed {result.interviewed_count} agents (dual platform)")
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Clean agent response: strip JSON tool call wrapper and extract actual text content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load simulation agent persona files"""
        import os
        import csv

        # Build path to persona files
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # First try reading Reddit JSON format
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"from reddit_profiles.json Load {len(profiles)} persona")
                return profiles
            except Exception as e:
                logger.warning(f"read reddit_profiles.json failed: {e}")
        
        # Fall back to reading Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV row format to unified profile format
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown"
                        })
                logger.info(f"from twitter_profiles.csv Load {len(profiles)} persona")
                return profiles
            except Exception as e:
                logger.warning(f"read twitter_profiles.csv failed: {e}")
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        Use LLM to select the most suitable agents to interview
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: list of complete agent information
                - selected_indices: list of agent indices (for API call)
                - reasoning: explanation of selection reasons
        """
        
        # Build agent summary list
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"agents_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """You are a professional interview planning expert. Based on the interview requirement, select the most suitable interview subjects from the simulation agents list.

Selection criteria:
1. Agent identity/profession should be related to the interview topic
2. Agents should hold unique or valuable viewpoints
3. Select a variety of perspectives (e.g. supporting, opposing, neutral, professional, etc.)
4. Prefer agents that are directly related to the event

Return JSON format:
{
    "selected_indices": [list of agent indices],
    "reasoning": "explanation of selection reasons"
}"""

        user_prompt = f"""Interview requirement:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else ""}

Available agents ({len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select at most {max_agents} of the most suitable interview agents, and explain your selection reasons."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Automatically selected by relevance")
            
            # Retrieve the full profile information for selected agents
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(f"LLM agent selection failed, using defaults: {e}")
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Use LLM to generate interview questions"""
        
        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]
        
        system_prompt = """You are a professional interviewer. Based on the interview requirement, generate 3-5 interview questions.

Question requirements:
1. Questions should be open-ended and detailed
2. Each side should be able to express their viewpoints
3. Cover facts, viewpoints, and emotional perspectives
4. Tailored to the specific interviewees
5. Each question should be under 50 words
6. Include necessary background context where appropriate

Return JSON format: {"questions": ["question1", "question2", ...]}"""

        user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else ""}

Interview subjects: {', '.join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"What is your perspective on {interview_requirement}?"])
            
        except Exception as e:
            logger.warning(f"Failed to generate interview questions: {e}")
            return [
                f"What is your view on {interview_requirement}?",
                "Which group do you represent, and what is their main stance?",
                "What suggestions do you have to address or improve this situation?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[agentsInterview],
        interview_requirement: str
    ) -> str:
        """Generate interview summary"""

        if not interviews:
            return "No interviews were completed."

        # Compile all interview content
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name}({interview.agent_role})]\n{interview.response[:500]}")

        system_prompt = """You are a professional analyst. Based on the interview content, generate a comprehensive interview summary.

Summary requirements:
1. Cover the main viewpoints of each side
2. Highlight contrasting perspectives
3. Identify the most valuable insights
4. Keep it focused; one viewpoint per section
5. Maximum 1000 characters

Format (required):
- Use plain prose
- Do not use markdown headings (#, ##, ###)
- Use dividers (---) where appropriate
- Use **bold** to highlight keywords"""

        user_prompt = f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate the interview summary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate interview summary: {e}")
            return f"Interviewed {len(interviews)} agents: " + ", ".join([i.agent_name for i in interviews])

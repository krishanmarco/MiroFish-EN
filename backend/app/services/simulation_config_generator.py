"""
Intelligent Simulation Configuration Generator
Use LLM to automatically generate detailed simulation parameters based on simulation requirements, document content, and graph information
Implement full automation without manual parameter setting

Adopt a step-by-step generation strategy to avoid failure from generating too long content at once:
1. Generate time configuration
2. Generate event configuration
3. Generate Agent configuration in batches
4. Generate platform configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')

# ChinaScheduleTimeConfiguration（BeijingTime）
CHINA_TIMEZONE_CONFIG = {
    # Late NightPeriod（AlmostNo OneActivity）
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # MorningPeriod（GraduallyWake）
    "morning_hours": [6, 7, 8],
    # WorkPeriod
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # EveningPeak（MostActive）
    "peak_hours": [19, 20, 21, 22],
    # Period（Active）
    "night_hours": [23],
    # ActiveCoefficient
    "activity_multipliers": {
        "dead": 0.05,      # AlmostNo One
        "morning": 0.4,    # MorningGraduallyActive
        "work": 0.7,       # WorkPeriod
        "peak": 1.5,       # EveningPeak
        "night": 0.5       # Late Night
    }
}


@dataclass
class AgentActivityConfig:
    """IndividualAgentActivityConfiguration"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # ActiveConfiguration (0.0-1.0)
    activity_level: float = 0.5  # OverallActive
    
    # StatementFrequency（Per HourExpectedStatementCount）
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # ActiveTime（24Hour Format，0-23）
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # ResponseSpeed（HotEventReactionDelay，Unit：SimulationMinute）
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # EmotionTendency (-1.01.0，)
    sentiment_bias: float = 0.0
    
    # Position（SpecificTopic）
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # InfluenceWeight（DetermineStatementAgentProbability）
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """TimeSimulationConfiguration（ChinaScheduleHabit）"""
    # SimulationTotalLength（SimulationHours）
    total_simulation_hours: int = 72  # Simulation72Hours（3Days）
    
    # Per RoundRepresentativeTime（SimulationMinute）- 60Minute（1Hours），TimeFlow
    minutes_per_round: int = 60
    
    # Per HourAgentQuantityRange
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # PeakPeriod（Evening19-22，ChinaMostActiveTime）
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # Period（0-5，AlmostNo OneActivity）
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Active
    
    # MorningPeriod
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # WorkPeriod
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """EventConfiguration"""
    # InitialEvent（SimulationStartTriggerEvent）
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # ScheduledEvent（SpecificTimeTriggerEvent）
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # HotTopicKeyword
    hot_topics: List[str] = field(default_factory=list)
    
    # OpinionGuideDirection
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """PlatformSpecificConfiguration"""
    platform: str  # twitter or reddit
    
    # RecommendAlgorithmWeight
    recency_weight: float = 0.4  # TimeFreshness
    popularity_weight: float = 0.3  # Popularity
    relevance_weight: float = 0.3  # Relevance
    
    # ViralSpreadThreshold（InteractionTriggerDissemination）
    viral_threshold: int = 10
    
    # （）
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """CompleteSimulationParameterConfiguration"""
    # Information
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # TimeConfiguration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # AgentConfigurationList
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # EventConfiguration
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # PlatformConfiguration
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLMConfiguration
    llm_model: str = ""
    llm_base_url: str = ""
    
    # Generate
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM
    
    def to_dict(self) -> Dict[str, Any]:
        """ConvertToDictionary"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """ConvertToJSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    SimulationConfigurationGenerate
    
    LLMSimulationRequirement、DocumentContent、GraphEntityInformation，
    Auto-generateMostSimulationParameterConfiguration
    
    AdoptStep-by-stepGenerateStrategy：
    1. GenerateTimeConfigurationEventConfiguration（）
    2. BatchGenerateAgentConfiguration（Batch10-20）
    3. GeneratePlatformConfiguration
    """
    
    # ContextMost
    MAX_CONTEXT_LENGTH = 50000
    # BatchGenerateAgentQuantity
    AGENTS_PER_BATCH = 15
    
    # ContextLength（）
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # TimeConfiguration
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # EventConfiguration
    ENTITY_SUMMARY_LENGTH = 300          # EntitySummary
    AGENT_SUMMARY_LENGTH = 300           # AgentConfigurationEntitySummary
    ENTITIES_PER_TYPE_DISPLAY = 20       # EntityQuantity
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        GenerateCompleteSimulationConfiguration（Step-by-stepGenerate）
        
        Args:
            simulation_id: SimulationID
            project_id: ID
            graph_id: GraphID
            simulation_requirement: SimulationRequirement
            document_text: DocumentContent
            entities: EntityList
            enable_twitter: Twitter
            enable_reddit: Reddit
            progress_callback: (current_step, total_steps, message)
            
        Returns:
            SimulationParameters: CompleteSimulationParameter
        """
        logger.info(f"StartGenerateSimulationConfiguration: simulation_id={simulation_id}, Entity={len(entities)}")
        
        # Total
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # TimeConfiguration + EventConfiguration + NBatchAgent + PlatformConfiguration
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # 1. ContextInformation
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== 1: GenerateTimeConfiguration ==========
        report_progress(1, "GenerateTimeConfiguration...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"TimeConfiguration: {time_config_result.get('reasoning', 'Success')}")

        # ========== 2: GenerateEventConfiguration ==========
        report_progress(2, "GenerateEventConfigurationHotTopic...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"EventConfiguration: {event_config_result.get('reasoning', 'Success')}")
        
        # ========== 3-N: BatchGenerateAgentConfiguration ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                f"GenerateAgentConfiguration ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"AgentConfiguration: Generate {len(all_agent_configs)} ")
        
        # ========== ToInitial Agent ==========
        logger.info("ToInitial Agent...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Initial: {assigned_count} ")
        
        # ========== Most: GeneratePlatformConfiguration ==========
        report_progress(total_steps, "GeneratePlatformConfiguration...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # MostParameter
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"SimulationConfigurationGenerateComplete: {len(params.agent_configs)} AgentConfiguration")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """LLMContext，MostLength"""
        
        # EntitySummary
        entity_summary = self._summarize_entities(entities)
        
        # Context
        context_parts = [
            f"## SimulationRequirement\n{simulation_requirement}",
            f"\n## EntityInformation ({len(entities)})\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # 500
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(Document)"
            context_parts.append(f"\n## DocumentContent\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """GenerateEntitySummary"""
        lines = []
        
        # Type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            # ConfigurationQuantitySummaryLength
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ...  {len(type_entities) - display_count} ")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """WithRetryLLMCall，ContainJSON"""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Retry
                    # Setmax_tokens，LLM
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # 
                if finish_reason == 'length':
                    logger.warning(f"LLM (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                # JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONFailed (attempt {attempt+1}): {str(e)[:80]}")
                    
                    # JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLMCallFailed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLMCallFailed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """JSON"""
        content = content.strip()
        
        # 
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # 
        if content and content[-1] not in '",}]':
            content += '"'
        
        # 
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """ConfigurationJSON"""
        import re
        
        # 
        content = self._fix_truncated_json(content)
        
        # JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                # All
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """GenerateTimeConfiguration"""
        # ConfigurationContextLength
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        
        # Most（80%agent）
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""Based on the simulation requirement below, generate a time simulation configuration.

{context_truncated}

## Task
Generate a JSON time configuration for the simulation.

### Recommended daily activity schedule:
- 0-5: Almost no activity (activity coefficient ~0.05)
- 6-8: Gradually increasing activity (activity coefficient ~0.4)
- 9-18: Work hours, moderate activity (activity coefficient ~0.7)
- 19-22: Evening peak, highest activity (activity coefficient ~1.5)
- 23: Winding down (activity coefficient ~0.5)
- Adjust peak/off-peak periods based on the event type and participant group:
  - Student groups: peak at 21-23; active all day; institutions active during work hours
  - Viral/controversial events may shift activity later into the night (off_peak_hours)

### Output JSON format (no markdown):

{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Explanation of why these time settings fit the event"
}}

Field descriptions:
- total_simulation_hours (int): Total simulation duration in hours, range 24-168; should match the event or topic lifespan
- minutes_per_round (int): Minutes per simulation round, range 30-120; default 60
- agents_per_hour_min (int): Minimum agents active per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Maximum agents active per hour (range: 1-{max_agents_allowed})
- peak_hours (list of int): Peak activity hours; should align with event/group patterns
- off_peak_hours (list of int): Low activity hours; typically late night
- morning_hours (list of int): Morning period hours
- work_hours (list of int): Work period hours
- reasoning (string): Explanation for the chosen configuration"""

        system_prompt = "You are a simulation configuration expert. Generate output in valid JSON format only. Configure time slots to reflect realistic daily activity patterns."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"TimeConfigurationLLMGenerateFailed: {e}, Configuration")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """GetTimeConfiguration（ChinaSchedule）"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # Per Round1Hours，TimeFlow
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Default schedule configuration (60 minutes per round)"
        }
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """TimeConfigurationResult，agents_per_hourTotalagent"""
        # Get
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        # ：Totalagent
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total agent count ({num_entities}), clamping.")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total agent count ({num_entities}), clamping.")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # ensure min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= agents_per_hour_max, adjusted to {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # Per Round1Hours
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # AlmostNo One
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """GenerateEventConfiguration"""
        
        # GetEntityTypeList， LLM 
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        # ToTypeRepresentativeEntity
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        # ConfigurationContextLength
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""Based on the simulation requirement below, generate an event configuration.

Simulation Requirement: {simulation_requirement}

{context_truncated}

## Available Entity Types
{type_info}

## Task
Generate an event configuration JSON with the following:
- Hot topic keywords relevant to the simulation
- A narrative/opinion direction for the event
- Initial posts to seed the simulation. **Each post MUST include a poster_type matching one of the entity types above.**

**Important**: The poster_type field MUST be one of the entity types listed above. Each initial post will be assigned to an agent of that type. For example: use "Official" or "University" type for institutional posts, "MediaOutlet" for media, "Student" for student posts.

Output JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<describe the opinion/narrative direction of the event>",
    "initial_posts": [
        {{"content": "Post content here", "poster_type": "EntityType (must match an available type)"}},
        ...
    ],
    "reasoning": "<explanation of why this event configuration fits the simulation>"
}}"""

        system_prompt = "You are a social media simulation expert. Generate valid JSON output only. The poster_type in each initial post MUST match one of the provided entity types."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"EventConfigurationLLMGenerateFailed: {e}, Configuration")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default configuration"
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """EventConfigurationResult"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        ToInitial Agent
        
         poster_type Most agent_id
        """
        if not event_config.initial_posts:
            return event_config
        
        # EntityType agent 
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        # Type（ LLM Format）
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }
        
        # Type agent ，Avoid agent
        used_indices: Dict[str, int] = {}
        
        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            
            # Found agent
            matched_agent_id = None
            
            # 1. 
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. 
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            # 3. Found，InfluenceMost agent
            if matched_agent_id is None:
                logger.warning(f"No matching agent found for type '{poster_type}', falling back to highest-influence agent.")
                if agent_configs:
                    # Influence，InfluenceMost
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"Initial: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """BatchGenerateAgentConfiguration"""
        
        # EntityInformation（ConfigurationSummaryLength）
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Based on the simulation requirement and entity list below, generate an activity configuration for each agent.

Simulation Requirement: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Activity Guidelines by Entity Type
Use the following as reference ranges when generating each agent's configuration:
- **Institutions** (University/Government): activity_level 0.1-0.3, active during work hours (9-17), response_delay 60-240 min, influence_weight 2.5-3.0
- **Media Outlets**: activity_level 0.4-0.6, active most of the day (8-23), response_delay 5-30 min, influence_weight 2.0-2.5
- **Individuals** (Student/Person/Alumni): activity_level 0.6-0.9, most active evenings (18-23), response_delay 1-15 min, influence_weight 0.8-1.2
- **Organizations/Groups**: activity_level 0.4-0.6, influence_weight 1.5-2.0
- General daily pattern: near-zero activity 0-5, gradually active 6-8, moderate 9-18, peak 19-22

Output JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <integer, must match entity index>,
            "activity_level": <float 0.0-1.0>,
            "posts_per_hour": <float, posting frequency>,
            "comments_per_hour": <float, comment frequency>,
            "active_hours": [<list of active hour integers>],
            "response_delay_min": <int, minimum response delay in minutes>,
            "response_delay_max": <int, maximum response delay in minutes>,
            "sentiment_bias": <float -1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <float, influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social simulation agent configuration expert. Generate valid JSON only. Configure each agent's activity parameters to reflect realistic behavior patterns for their entity type."
        
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"AgentConfigurationBatchLLMGenerateFailed: {e}, Generate")
            llm_configs = {}
        
        # AgentActivityConfigObject
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            # LLMGenerate，Generate
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """GenerateIndividualAgentConfiguration（ChinaSchedule）"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo"]:
            # Institution：WorkTimeActivity，Frequency，Influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # ：DaysActivity，Frequency，Influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # /：Work+EveningActivity，Frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # ：EveningTo，Frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # +Evening
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # ：EveningTo
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # +Evening
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # ：EveningPeak
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Days+Evening
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    


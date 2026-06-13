from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

GRAPH_CONTEXT_TEMPLATE = """
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
"""

CHUNK_CONTEXT_TEMPLATE = """
-----Sources-----
```csv
{text_units_context}
```
"""

MM_PROMPTS: dict[str, Any] = {}

MM_PROMPTS["mm_entity_extraction"] = """-Goal-
Analyze the provided text section, image, and scene graph associated with the image to create a multimodal knowledge graph with the entity-relationship diagram.
Use {language} as the output language.

The scene graph provides the object and relation information in the image, which is formatted as:
```
- <object_0>: <object_category>, <object_bbox>
- <object_1>: <object_category>, <object_bbox>
...
- <relation_0>: <object_0> <relation_name> <object_1>
- <relation_2>: <object_1> <relation_name> <object_3>
...
```
The <object_bbox> is the bounding boxes of each object region, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

-Steps-
1. **Entity Extraction**: Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use the same language as the input text. If English, capitalize the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. **Relation Extraction**: From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity, ranging from 0 to 10
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. **Visual-Scene Mapping**: Based on the provided image and scene graph, map visual objects/relations in the image to text-derived entities/relationships.

3.1 Identify the text entity that is most relevant to the overall image and extract the following information:
- entity_name: the name of the entity that best represents the overall image. Please use the extracted entity name from step 1, not the object category from the scene graph.
- strength: a numeric score indicating the strength of the match, ranging from 0 to 10
Format the image mapping as ("mapping"{tuple_delimiter}"<image>"{tuple_delimiter}<entity_name>{tuple_delimiter}<strength>)

3.2 For each object in the scene graph, if the object visually depicts a text entity identified in step 1, extract the following information:
- object_id: the id of the object in the scene graph
- entity_name: the name of the entity it represents. Please use the extracted entity name from step 1, not the object category from the scene graph.
- strength: a numeric score indicating the strength of the match, ranging from 0 to 10
Format each object mapping as ("mapping"{tuple_delimiter}<object_id>{tuple_delimiter}<entity_name>{tuple_delimiter}<strength>)

3.3 For each relation in the scene graph, if the relation visually represents a text relationship identified in step 2, extract the following information:
- relation_id: the id of the relation in the scene graph
- source_entity: the source entity of the relationship it represents. Please use the extracted entity name from step 1, not the object category from the scene graph.
- target_entity: the target entity of the relationship it represents. Please use the extracted entity name from step 1, not the object category from the scene graph.
- strength: a numeric score indicating the strength of the match, ranging from 0 to 10
Format each relation mapping as ("mapping"{tuple_delimiter}<relation_id>{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<strength>)

3.4 For those objects or relations without a corresponding text entity or relationship, please ignore them.

4. Return output as a single list of all the entities, relationships, object mappings, and relation mappings identified in steps 1-3. Use {record_delimiter} as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: [{entity_types}]
Text: {input_text}

{image_desc}
Scene Graph: 
```
{scene_graph}
```
######################
Output:
"""

MM_PROMPTS["mm_examples"] = [
    """Example 1:

Entity_types: [event,place,organisation,building,location]
Text: Mount Fuji, an active stratovolcano situated on Japan's Honshu Island, rises to a summit elevation of 3,776.24 meters. Its strikingly symmetrical cone, adorned with snow for approximately five months each year, is a breathtaking natural wonder.
As a cherished cultural icon of Japan, Mount Fuji is frequently celebrated in art and photography and draws countless visitors, including sightseers, hikers, and climbers. Notably, Mount Fuji, alongside cherry blossoms and the Shinkansen—colloquially known as the bullet train, a network of high-speed railway lines in Japan—is celebrated as one of the country's most iconic national symbols.

Image Description: 
Mount Fuji and the Shinkansen electric car passing in front of it.
Scene Graph: 
```
- <object_0>: train, (0.06, 0.64, 1.0, 0.77)
- <object_1>: fence, (0.0, 0.8, 0.98, 0.88)
- <object_2>: snow, (0.25, 0.29, 0.67, 0.49)
- <object_3>: mountain, (0.0, 0.3, 1.0, 0.64)
- <relation_0>: <object_0> over <object_1>
- <relation_1>: <object_2> on <object_3>
- <relation_2>: <object_3> behind <object_0>
```
################
Output:
("entity"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"location"{tuple_delimiter}"Mount Fuji is an active stratovolcano located on Japan's Honshu Island, with a peak elevation of 3,776.24 meters."){record_delimiter}
("entity"{tuple_delimiter}"HONSHU ISLAND"{tuple_delimiter}"location"{tuple_delimiter}"Honshu Island is the largest island of Japan, where Mount Fuji is situated."){record_delimiter}
("entity"{tuple_delimiter}"CHERRY BLOSSOMS"{tuple_delimiter}"concept"{tuple_delimiter}"Cherry blossoms are a symbol of Japan, known for their beauty and cultural significance, often associated with the arrival of spring."){record_delimiter}
("entity"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}"technology"{tuple_delimiter}"The Shinkansen, also known as the bullet train, is a network of high-speed railway lines in Japan."){record_delimiter}
("relationship"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"HONSHU ISLAND"{tuple_delimiter}"Mount Fuji is located on Honshu Island, making the island its geographical setting."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"CHERRY BLOSSOMS"{tuple_delimiter}"Both Mount Fuji and cherry blossoms are iconic symbols of Japan, often celebrated together in cultural contexts."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}"Mount Fuji and the Shinkansen are both recognized as national symbols of Japan."{tuple_delimiter}7){record_delimiter}
("mapping"{tuple_delimiter}"<image>"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}8){record_delimiter}
("mapping"{tuple_delimiter}"<object_3>"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}9){record_delimiter}
("mapping"{tuple_delimiter}"<object_0>"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}7){record_delimiter}
("mapping"{tuple_delimiter}"<relation_2>"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}7){completion_delimiter}
#############################"""
]

MM_PROMPTS["mapping_extract_prompt"] = """-Goal-
Based on the provided image, visual scene graph, and textual entities and relationships, map visual objects/relations in the image to textual entities/relationships.
Use {language} as the output language.

-Input Format-
Each textual entity are formatted as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>), which contains the following information:
- entity_name: Name of the entity.
- entity_type: Name of the entity type.
- entity_description: Comprehensive description of the entity's attributes and activities.

Each textual relationship are formatted as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>), which contains the following information:
- source_entity: name of the source entity, as defined in the textual entities.
- target_entity: name of the target entity, as defined in the textual entities.
- relationship_description: explanation as to why the source entity and the target entity are related to each other.
- relationship_strength: a numeric score indicating the strength of the relationship between the source entity and target entity, ranging from 0 to 10.

The scene graph provides the object and relation information in the image, which is formatted as:
```
- <object_0>: <object_category>, <object_bbox>
- <object_1>: <object_category>, <object_bbox>
...
- <relation_0>: <object_0> <relation_name> <object_1>
- <relation_2>: <object_1> <relation_name> <object_3>
...
```
The <object_bbox> is the bounding boxes of each object region, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

-Steps-
1. Identify the textual entity that is most relevant to the overall image and extract the following information:
- entity_name: the name of the entity that best represents the overall image. Please use the provided entity name from the input data, not the object category from the scene graph.
- strength: a numeric score indicating the strength of the match, ranging from 0 to 10
Format the image mapping as ("mapping"{tuple_delimiter}"<image>"{tuple_delimiter}<entity_name>{tuple_delimiter}<strength>)

2. For each object in the scene graph, if the object visually depicts a textual entity identified in the input data, extract the following information:
- object_id: the id of the object in the scene graph
- entity_name: the name of the entity it represents. Please use the provided entity name from the input data, not the object category from the scene graph.
- strength: a numeric score indicating the strength of the match, ranging from 0 to 10
Format each object mapping as ("mapping"{tuple_delimiter}<object_id>{tuple_delimiter}<entity_name>{tuple_delimiter}<strength>)

3. For each relation in the scene graph, if the relation visually represents a textual relationship identified in the input data, extract the following information:
- relation_id: the id of the relation in the scene graph
- source_entity: the source entity of the relationship it represents. Please use the provided entity name from the input data, not the object category from the scene graph.
- target_entity: the target entity of the relationship it represents. Please use the provided entity name from the input data, not the object category from the scene graph.
- strength: a numeric score indicating the strength of the match, ranging from 0 to 10
Format each relation mapping as ("mapping"{tuple_delimiter}<relation_id>{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<strength>)

4. For those objects or relations without a corresponding text entity or relationship, please ignore them.

5. Return output as a single list of mappings identified in steps 1-3. Use {record_delimiter} as the list delimiter.

6. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Textual Entities: 
{entities}
Textual Relationships:
{relationships}

{image_desc}
Scene Graph: 
```
{scene_graph}
```
######################
Output:
"""

MM_PROMPTS["mapping_examples"] = [
    """Example 1:
Textual Entities: 
("entity"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"location"{tuple_delimiter}"Mount Fuji is an active stratovolcano located on Japan's Honshu Island, with a peak elevation of 3,776.24 meters."){record_delimiter}
("entity"{tuple_delimiter}"HONSHU ISLAND"{tuple_delimiter}"location"{tuple_delimiter}"Honshu Island is the largest island of Japan, where Mount Fuji is situated."){record_delimiter}
("entity"{tuple_delimiter}"CHERRY BLOSSOMS"{tuple_delimiter}"concept"{tuple_delimiter}"Cherry blossoms are a symbol of Japan, known for their beauty and cultural significance, often associated with the arrival of spring."){record_delimiter}
("entity"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}"technology"{tuple_delimiter}"The Shinkansen, also known as the bullet train, is a network of high-speed railway lines in Japan.")
Textual Relationships:
("relationship"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"HONSHU ISLAND"{tuple_delimiter}"Mount Fuji is located on Honshu Island, making the island its geographical setting."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"CHERRY BLOSSOMS"{tuple_delimiter}"Both Mount Fuji and cherry blossoms are iconic symbols of Japan, often celebrated together in cultural contexts."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}"Mount Fuji and the Shinkansen are both recognized as national symbols of Japan."{tuple_delimiter}7)

Image Description: 
Mount Fuji and the Shinkansen electric car passing in front of it.
Scene Graph: 
```
- <object_0>: train, (0.06, 0.64, 1.0, 0.77)
- <object_1>: fence, (0.0, 0.8, 0.98, 0.88)
- <object_2>: snow, (0.25, 0.29, 0.67, 0.49)
- <object_3>: mountain, (0.0, 0.3, 1.0, 0.64)
- <relation_0>: <object_0> over <object_1>
- <relation_1>: <object_2> on <object_3>
- <relation_2>: <object_3> behind <object_0>
```
################
Output:
("mapping"{tuple_delimiter}"<image>"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}8){record_delimiter}
("mapping"{tuple_delimiter}"<object_3>"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}9){record_delimiter}
("mapping"{tuple_delimiter}"<object_0>"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}7){record_delimiter}
("mapping"{tuple_delimiter}"<relation_2>"{tuple_delimiter}"MOUNT FUJI"{tuple_delimiter}"SHINKANSEN"{tuple_delimiter}7){completion_delimiter}
#############################""",
    """Example 2:
Textual Entities:
("entity"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}"location"{tuple_delimiter}"Whitehaven Beach is a pristine silica-sand beach in Australia's Whitsunday Islands, known for its turquoise waters."){record_delimiter}
("entity"{tuple_delimiter}"HILL INLET"{tuple_delimiter}"location"{tuple_delimiter}"Hill Inlet is a tidal estuary at the northern end of Whitehaven Beach, famous for its swirling sand patterns."){record_delimiter}
("entity"{tuple_delimiter}"MARINE LIFE"{tuple_delimiter}"concept"{tuple_delimiter}"The Great Barrier Reef ecosystem near Whitehaven Beach hosts diverse marine species like turtles and tropical fish."){record_delimiter}
("entity"{tuple_delimiter}"ECO-TOURISM"{tuple_delimiter}"concept"{tuple_delimiter}"Eco-tourism initiatives at Whitehaven Beach emphasize sustainable visitor experiences and environmental preservation.")
Textual Relationships:
("relationship"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}"HILL INLET"{tuple_delimiter}"Hill Inlet forms part of Whitehaven Beach's unique geography, creating iconic sand-and-water mosaics."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}"MARINE LIFE"{tuple_delimiter}"The beach's proximity to coral reefs supports rich marine biodiversity."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}"ECO-TOURISM"{tuple_delimiter}"Whitehaven Beach serves as a model for eco-tourism with strict conservation measures."{tuple_delimiter}8)

Image Description:
Aerial view of Whitehaven Beach showing swirling sand patterns at Hill Inlet, with snorkelers near a coral reef.
Scene Graph:
```
- <object_0>: sand, (0.1, 0.15, 0.9, 0.45)
- <object_1>: water, (0.0, 0.4, 1.0, 0.7)
- <object_2>: man, (0.65, 0.75, 0.85, 0.85)
- <object_3>: coral, (0.5, 0.8, 0.95, 0.95)
- <relation_0>: <object_1> surrounding <object_0>
- <relation_1>: <object_2> near <object_3>
- <relation_2>: <object_0> patterned_with <object_1>
```
################
Output:
("mapping"{tuple_delimiter}"<image>"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}9){record_delimiter}
("mapping"{tuple_delimiter}"<object_0>"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}8){record_delimiter}
("mapping"{tuple_delimiter}"<object_2>"{tuple_delimiter}"ECO-TOURISM"{tuple_delimiter}7){record_delimiter}
("mapping"{tuple_delimiter}"<object_3>"{tuple_delimiter}"MARINE LIFE"{tuple_delimiter}8){record_delimiter}
("mapping"{tuple_delimiter}"<relation_2>"{tuple_delimiter}"WHITEHAVEN BEACH"{tuple_delimiter}"HILL INLET"{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 3:
Textual Entities:
("entity"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}"landmark"{tuple_delimiter}"A basilica in Barcelona designed by Antoni Gaudí, known for its intricate modernist architecture."){record_delimiter}
("entity"{tuple_delimiter}"GAUDÍ"{tuple_delimiter}"person"{tuple_delimiter}"Catalan architect whose organic style defines Barcelona's cityscape."){record_delimiter}
("entity"{tuple_delimiter}"TRENCADÍS"{tuple_delimiter}"art"{tuple_delimiter}"A mosaic technique using broken ceramic tiles, frequently used in Gaudí's designs."){record_delimiter}
("entity"{tuple_delimiter}"BARCELONA"{tuple_delimiter}"location"{tuple_delimiter}"A Spanish city where Sagrada Familia serves as a cultural and architectural icon.")
Textual Relationships:
("relationship"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}"GAUDÍ"{tuple_delimiter}"Gaudí dedicated his final years to designing Sagrada Familia, which remains unfinished."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}"TRENCADÍS"{tuple_delimiter}"The basilica's spires feature trencadís mosaics that shimmer in sunlight."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}"BARCELONA"{tuple_delimiter}"The Sagrada Familia is a central landmark in Barcelona's urban identity."{tuple_delimiter}9)

Image Description:
Close-up of Sagrada Familia's spires with colorful trencadís details, against a Barcelona city backdrop.
Scene Graph:
```
- <object_0>: building, (0.1, 0.1, 0.9, 0.95)
- <object_1>: mosaic, (0.4, 0.7, 0.6, 0.85)
- <object_2>: city, (0.0, 0.0, 1.0, 0.2)
- <relation_0>: <object_1> on <object_0>
- <relation_1>: <object_0> towering_above <object_2>
```
################
Output:
("mapping"{tuple_delimiter}"<image>"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}9){record_delimiter}
("mapping"{tuple_delimiter}"<object_0>"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}9){record_delimiter}
("mapping"{tuple_delimiter}"<object_1>"{tuple_delimiter}"TRENCADÍS"{tuple_delimiter}8){record_delimiter}
("mapping"{tuple_delimiter}"<relation_1>"{tuple_delimiter}"SAGRADA FAMILIA"{tuple_delimiter}"BARCELONA"{tuple_delimiter}9){completion_delimiter}
#############################"""
]

MM_PROMPTS["entity_format"] = """("entity"{tuple_delimiter}"{entity_name}"{tuple_delimiter}"{entity_type}"{tuple_delimiter}"{entity_description}")"""
MM_PROMPTS["relationship_format"] = """("relationship"{tuple_delimiter}"{source_entity}"{tuple_delimiter}"{target_entity}"{tuple_delimiter}"{relationship_description}"{tuple_delimiter}"{relationship_strength}")"""

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "<|RECORD|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["event", "place", "organisation", "building", "person", "species", "category"]

PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################""",
]

PROMPTS["summarize_entity_descriptions"] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entiti_continue_extraction"] = """MANY entities were missed in the last extraction.  Add them below using the same format:"""

PROMPTS["entiti_if_loop_extraction"] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added."""

PROMPTS["fail_response"] = ("Sorry, I'm not able to provide an answer to that question.[no-context]")

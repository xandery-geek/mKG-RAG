import json
from tools.utils import get_token_length

SECTION_DELIMITER = "<MMRAG_SECTION_DELIMITER>"
IMAGE_START = "<MMRAG_IMAGE_START>"
IMAGE_END = "<MMRAG_IMAGE_END>"

MARKDOWN_TEMPLATE = {}
MARKDOWN_TEMPLATE["section"] = "## {title}\n\n{text}\n"
MARKDOWN_TEMPLATE["image"] = "![{desc}]({url})\n"
MARKDOWN_TEMPLATE["image_marker"] = "{image_start}{images}{image_end}".format(
    image_start=IMAGE_START, image_end=IMAGE_END, images="{images}"
)

DEFAULT_INVALID_TITLES = ["notes", "references", "see also", "external links", 
                          "footnotes", "further reading", "topics"]


def get_document_url(data):
    return list(data.keys())[0]


def load_document_content(data, load_img=False, invalid_titles=None):
    """
    Load the document content from the data
    Args:
        data (dict): The data dictionary
        load_img (bool): Whether to load the images
        invalid_titles (list): The list of invalid section titles
    Returns:
        section_titles (list): The list of section titles
        section_texts (list): The list of section texts
        section_images (list): The list of image lists, each list contains the image urls and descriptions. 
                                If load_img is False, then each image list will be empty.
    """
    url = list(data.keys())[0]
    data = data[url]

    article_title = data["title"]
    section_titles = data["section_titles"]
    section_texts = data["section_texts"]
    section_images = [[] for _ in range(len(section_titles))]

    if load_img:
        image_urls = data["image_urls"]
        image_sec_indices = data["image_section_indices"]
        image_desc = data["image_reference_descriptions"]

        for idx, url, desc in zip(image_sec_indices, image_urls, image_desc):
            section_images[idx].append(
                {
                    "url": url,
                    "desc": desc
                }
            )
    
    if invalid_titles is not None:
        invalid_titles = set([title.lower() for title in invalid_titles])

        valid_titles, valid_texts, valid_images = [], [], []
        for title, text, images in zip(section_titles, section_texts, section_images):
            # filter out invalid titles
            if title.lower() not in invalid_titles:
                # append the article title to the section title if it is not already included
                if article_title.strip().lower() not in title.lower():
                    title = f"{article_title} - {title}"

                valid_titles.append(title)
                valid_texts.append(text)
                valid_images.append(images)
        return valid_titles, valid_texts, valid_images
    else:
        return section_titles, section_texts, section_images


def convert_to_markdown(data, 
                        load_img=False, 
                        invalid_titles=DEFAULT_INVALID_TITLES,
                        min_token_size=200,
                        max_token_size=500,
                        max_section_num=None):
    """
    Converts the data to markdown format
    Args:
        data (dict): The data dictionary
        load_img (bool): Whether to load the images
        invalid_titles (list): The list of invalid section titles
        min_token_size (int): The minimum token size for a section
        max_token_size (int): The maximum token size for a section
        max_section_num (int): The maximum number of sections
    """

    def insert_images(images):
        return MARKDOWN_TEMPLATE["image_marker"].format(images=json.dumps(images))
    
    def chunk_section(section, min_token_size, max_token_size):
        chunks = []
        lines = section.split("\n")
        
        chunk_token = 0
        chunk = ""
        for line in lines:
            line_token = get_token_length(line)
            if chunk_token + line_token > max_token_size:
                if chunk_token < min_token_size:
                    # if the chunk is too small, then append the line to the chunk
                    chunk += line + "\n"
                    chunks.append(chunk)
                    chunk = ""
                    chunk_token = 0
                else:
                    # if the chunk is large enough, then append the chunk to the chunks
                    chunks.append(chunk)
                    chunk = line + "\n"
                    chunk_token = line_token
            else:
                chunk += line + "\n"
                chunk_token += line_token
        
        if 0 < chunk_token < min_token_size:
            # if the last chunk is too small, then append the last chunk to the last chunk
            chunks[-1] += chunk
        elif chunk_token >= min_token_size:
            # if the last chunk is large enough, then append the last chunk to the chunks
            chunks.append(chunk)

        return chunks

    section_titles, section_texts, section_images \
          = load_document_content(data, load_img=load_img, invalid_titles=invalid_titles)
    
    section_delimiter = "\n" + SECTION_DELIMITER + "\n"
    
    pre_section, markdown = "", ""
    section_delimiter_cnt = 0
    for idx, (sec_title, sec_text, sec_images) in enumerate(zip(section_titles, section_texts, section_images)):
        if sec_text.strip() == "":
            continue

        section = MARKDOWN_TEMPLATE["section"].format(title=sec_title, text=sec_text)

        if len(sec_images) > 0:
            if pre_section != "":
                # The images are alined with the current section. When the current section has images, do not append the previous section
                markdown += pre_section + section_delimiter
                section_delimiter_cnt += 1

            token_size = get_token_length(section)
            if token_size > max_token_size:
                # If the section is large, then chunk the section and insert the images at the end of each chunk
                chunks = chunk_section(section, min_token_size, max_token_size)
                chunks = [chunk + insert_images(sec_images) for chunk in chunks]
                markdown += section_delimiter.join(chunks) + section_delimiter
                section_delimiter_cnt += len(chunks)
            else:
                # Even if the section is small, it should be a separate section if it has images
                section += insert_images(sec_images)
                markdown += section + section_delimiter
                section_delimiter_cnt += 1
        else:
            # Append the current section to the previous section if the previous section is small
            section = pre_section + section

            token_size = get_token_length(section)
            if token_size > max_token_size:
                # If the section is large, then chunk the section
                chunks = chunk_section(section, min_token_size, max_token_size)
                markdown += section_delimiter.join(chunks) + section_delimiter
                section_delimiter_cnt += len(chunks)
            elif token_size > min_token_size:
                markdown += section + section_delimiter
                section_delimiter_cnt += 1
            else:
                # Record small section for the next section
                pre_section = section
                continue

        pre_section = ""

        if max_section_num is not None and section_delimiter_cnt >= max_section_num:
            break
    
    # remove the last SECTION_DELIMITER
    if markdown.endswith(section_delimiter):
        markdown = markdown[:-len(section_delimiter)]
    
    return markdown
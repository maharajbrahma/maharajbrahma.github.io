---
layout: publications
title: Publications
description: Publications related to research work done
---

<div class="publication-list">
{% for p in site.data.publications %}
<article class="publication">
    <img class="thumb" src="{{ p.thumb }}" alt="{{ p.title }} thumbnail" />
    <div class="meta">
    <h2 class="title">{{ p.title }}</h2>
    <div class="authors">{{ p.authors }}. {% if p.venue %} {{ p.venue }}{% endif %} {{ p.year }}.</div>
    <div class="badges">
        {% if p.links.pdf %}<a class="badge pdf" href="{{ p.links.pdf }}">PDF</a>{% endif %}
        {% if p.links.abstract %}<a class="badge abstract" href="{{ p.links.abstract }}">Abstract</a>{% endif %}
        {% if p.links.bibtex %}<a class="badge bib" href="{{ p.links.bibtex }}">BibTeX</a>{% endif %}
        {% if p.links.slides %}<a class="badge slides" href="{{ p.links.slides }}">Slides</a>{% endif %}
        {% if p.links.poster %}<a class="badge poster" href="{{ p.links.poster }}">Poster</a>{% endif %}
        {% if p.links.dataset %}<a class="badge slides" href="{{ p.links.dataset }}">Dataset</a>{% endif %}
        {% if p.links.blog %}<a class="badge pdf" href="{{ p.links.blog }}">Blog</a>{% endif %}

    </div>
    </div>
</article>
{% endfor %}
</div>

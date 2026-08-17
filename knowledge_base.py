"""
Small sample knowledge base for the RAG demo.
Topic: solar panels — chosen because a realistic user query about it
naturally needs several DIFFERENT task types (explanation, summary,
key points, AND a math calculation) — which is exactly what lets us
demonstrate capability-mismatch routing later.
"""

DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "Solar panels convert sunlight into electricity using "
            "photovoltaic (PV) cells made of semiconductor materials, "
            "typically silicon. When sunlight hits a PV cell, it "
            "knocks electrons loose, creating a flow of electric "
            "current. Multiple cells are wired together into a panel, "
            "and multiple panels form an array."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Residential solar panel systems typically cost between "
            "$12,000 and $18,000 after installation, before any "
            "government incentives or tax credits are applied. Costs "
            "vary based on system size, roof type, and local labor "
            "rates."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "The main advantages of solar power include reduced "
            "electricity bills, low maintenance requirements once "
            "installed, a long operational lifespan of 25-30 years, "
            "and a reduction in household carbon emissions. Panels "
            "also tend to increase property resale value."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "Solar panel efficiency has improved significantly over "
            "the past decade, with most commercial panels now "
            "converting 18-22% of incoming sunlight into usable "
            "electricity. Panel output also depends on angle, "
            "shading, and local climate conditions."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "Homeowners can estimate their solar payback period by "
            "dividing the total system cost by their annual "
            "electricity bill savings. Most residential systems reach "
            "payback within 6 to 10 years, after which the "
            "electricity generated is effectively free."
        ),
    },
]

print("✅ parse_annotations.py loaded")

def extract_relevant_data(annotation_json):
    transcript = ""
    sentiments = []
    labels = []
    poses_summary = "Pose data summarization not yet implemented."

    try:
        for speech in annotation_json['annotationResults'][0].get('speechTranscriptions', []):
            for alt in speech.get('alternatives', []):
                transcript += alt.get('transcript', '') + "\n"

        for label in annotation_json['annotationResults'][0].get('segmentLabelAnnotations', []):
            labels.append(label['entity']['description'])

        sentiments.append("Sentiment extraction not available in this version.")

    except Exception as e:
        print("❌ Error while parsing annotations:", e)

    return {
        "transcript": transcript.strip(),
        "sentiments": sentiments,
        "labels": labels,
        "pose_summary": poses_summary
    }

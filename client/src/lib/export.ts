import { jsPDF } from "jspdf";
import type { Workout, Script, Reflection, ChatMessage } from "@shared/schema";

export function exportWorkoutAsPDF(workout: Workout) {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - (margin * 2);
  let y = margin;

  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  const titleLines = doc.splitTextToSize(workout.title, contentWidth);
  doc.text(titleLines, margin, y);
  y += titleLines.length * 8;

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100);
  doc.text(new Date(workout.timestamp).toLocaleString(), margin, y);
  y += 10;

  doc.setTextColor(0);
  workout.exercises.forEach((exercise, index) => {
    if (y > 270) {
      doc.addPage();
      y = margin;
    }

    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    const exerciseTitle = `${index + 1}. ${exercise.name}`;
    const exerciseTitleLines = doc.splitTextToSize(exerciseTitle, contentWidth);
    doc.text(exerciseTitleLines, margin, y);
    y += exerciseTitleLines.length * 6 + 2;

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    const details: string[] = [];
    if (exercise.sets) details.push(`Sets: ${exercise.sets}`);
    if (exercise.reps) details.push(`Reps: ${exercise.reps}`);
    if (exercise.duration) details.push(`Duration: ${exercise.duration}`);
    
    if (details.length > 0) {
      doc.text(details.join(" | "), margin + 5, y);
      y += 6;
    }

    if (exercise.notes) {
      doc.setTextColor(60);
      const notesLines = doc.splitTextToSize(exercise.notes, contentWidth - 5);
      doc.text(notesLines, margin + 5, y);
      y += notesLines.length * 5 + 3;
      doc.setTextColor(0);
    }

    y += 5;
  });

  doc.save(`workout-${new Date(workout.timestamp).toISOString().split('T')[0]}.pdf`);
}

export function exportWorkoutAsTXT(workout: Workout) {
  let text = `${workout.title}\n`;
  text += `Generated: ${new Date(workout.timestamp).toLocaleString()}\n`;
  text += `${"=".repeat(60)}\n\n`;

  workout.exercises.forEach((exercise, index) => {
    text += `${index + 1}. ${exercise.name}\n`;
    const details: string[] = [];
    if (exercise.sets) details.push(`Sets: ${exercise.sets}`);
    if (exercise.reps) details.push(`Reps: ${exercise.reps}`);
    if (exercise.duration) details.push(`Duration: ${exercise.duration}`);
    if (details.length > 0) {
      text += `   ${details.join(" | ")}\n`;
    }
    if (exercise.notes) {
      text += `   ${exercise.notes}\n`;
    }
    text += `\n`;
  });

  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `workout-${new Date(workout.timestamp).toISOString().split('T')[0]}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportScriptAsPDF(script: Script) {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - (margin * 2);
  let y = margin;

  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  const titleLines = doc.splitTextToSize(script.title, contentWidth);
  doc.text(titleLines, margin, y);
  y += titleLines.length * 8;

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100);
  doc.text(`Type: ${script.type}`, margin, y);
  y += 6;
  doc.text(new Date(script.timestamp).toLocaleString(), margin, y);
  y += 12;

  doc.setTextColor(0);
  doc.setFontSize(11);
  const contentLines = doc.splitTextToSize(script.content, contentWidth);
  
  contentLines.forEach((line: string) => {
    if (y > 270) {
      doc.addPage();
      y = margin;
    }
    doc.text(line, margin, y);
    y += 6;
  });

  doc.save(`script-${script.type.toLowerCase()}-${new Date(script.timestamp).toISOString().split('T')[0]}.pdf`);
}

export function exportScriptAsTXT(script: Script) {
  let text = `${script.title}\n`;
  text += `Type: ${script.type}\n`;
  text += `Generated: ${new Date(script.timestamp).toLocaleString()}\n`;
  text += `${"=".repeat(60)}\n\n`;
  text += script.content;

  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `script-${script.type.toLowerCase()}-${new Date(script.timestamp).toISOString().split('T')[0]}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportReflectionAsPDF(reflection: Reflection) {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - (margin * 2);
  let y = margin;

  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  const titleLines = doc.splitTextToSize(reflection.title, contentWidth);
  doc.text(titleLines, margin, y);
  y += titleLines.length * 8;

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100);
  doc.text(new Date(reflection.timestamp).toLocaleString(), margin, y);
  y += 12;

  doc.setTextColor(0);
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("Wins", margin, y);
  y += 6;
  doc.setFontSize(11);
  doc.setFont("helvetica", "normal");
  reflection.wins.forEach((win) => {
    const winLines = doc.splitTextToSize(`• ${win}`, contentWidth - 5);
    winLines.forEach((line: string) => {
      if (y > 270) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin + 5, y);
      y += 6;
    });
  });
  y += 4;

  if (y > 270) {
    doc.addPage();
    y = margin;
  }
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("Challenges", margin, y);
  y += 6;
  doc.setFontSize(11);
  doc.setFont("helvetica", "normal");
  reflection.challenges.forEach((challenge) => {
    const challengeLines = doc.splitTextToSize(`• ${challenge}`, contentWidth - 5);
    challengeLines.forEach((line: string) => {
      if (y > 270) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin + 5, y);
      y += 6;
    });
  });
  y += 4;

  if (y > 270) {
    doc.addPage();
    y = margin;
  }
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("Growth Areas", margin, y);
  y += 6;
  doc.setFontSize(11);
  doc.setFont("helvetica", "normal");
  reflection.growth.forEach((growthItem) => {
    const growthLines = doc.splitTextToSize(`• ${growthItem}`, contentWidth - 5);
    growthLines.forEach((line: string) => {
      if (y > 270) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin + 5, y);
      y += 6;
    });
  });

  doc.save(`reflection-${new Date(reflection.timestamp).toISOString().split('T')[0]}.pdf`);
}

export function exportReflectionAsTXT(reflection: Reflection) {
  let text = `${reflection.title}\n`;
  text += `Generated: ${new Date(reflection.timestamp).toLocaleString()}\n`;
  text += `${"=".repeat(60)}\n\n`;
  
  text += `WINS\n`;
  text += `${"-".repeat(60)}\n`;
  reflection.wins.forEach((win, index) => {
    text += `${index + 1}. ${win}\n`;
  });
  text += `\n`;

  text += `CHALLENGES\n`;
  text += `${"-".repeat(60)}\n`;
  reflection.challenges.forEach((challenge, index) => {
    text += `${index + 1}. ${challenge}\n`;
  });
  text += `\n`;

  text += `GROWTH AREAS\n`;
  text += `${"-".repeat(60)}\n`;
  reflection.growth.forEach((growthItem, index) => {
    text += `${index + 1}. ${growthItem}\n`;
  });

  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `reflection-${new Date(reflection.timestamp).toISOString().split('T')[0]}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportChatAsPDF(messages: ChatMessage[]) {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - (margin * 2);
  let y = margin;

  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  doc.text("Chat Conversation with Spirit", margin, y);
  y += 10;

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100);
  doc.text(`Exported: ${new Date().toLocaleString()}`, margin, y);
  y += 12;

  doc.setTextColor(0);
  messages.forEach((message) => {
    if (y > 260) {
      doc.addPage();
      y = margin;
    }

    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(message.role === "user" ? 80 : 0);
    doc.text(message.role === "user" ? "You" : "Spirit", margin, y);
    y += 6;

    doc.setFont("helvetica", "normal");
    const contentLines = doc.splitTextToSize(message.content, contentWidth);
    contentLines.forEach((line: string) => {
      if (y > 270) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += 5;
    });
    
    y += 6;
  });

  doc.save(`chat-${new Date().toISOString().split('T')[0]}.pdf`);
}

export function exportChatAsTXT(messages: ChatMessage[]) {
  let text = `Chat Conversation with Spirit\n`;
  text += `Exported: ${new Date().toLocaleString()}\n`;
  text += `${"=".repeat(60)}\n\n`;

  messages.forEach((message) => {
    text += `${message.role === "user" ? "You" : "Spirit"}:\n`;
    text += `${message.content}\n\n`;
  });

  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chat-${new Date().toISOString().split('T')[0]}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

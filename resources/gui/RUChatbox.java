package robotuniverse;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.font.FontRenderContext;
import java.awt.image.BufferedImage;

/**
 * @author Roberto
 */
public class RUChatbox implements MouseWheelListener {
    
    public class tab
    {
        public tab()
        {
        	object.addMouseListener(new MouseAdapter() { public void mouseEntered(MouseEvent evt) { tabMouseOver(); } });
        	object.addMouseListener(new MouseAdapter() { public void mouseExited(MouseEvent evt) { tabMouseOut(); } });
        	object.addMouseListener(new MouseAdapter() { public void mouseClicked(MouseEvent evt) { tabClicked(evt); } });
        }
    	
    	public int status=0;
        /*
         * 0 - not select
         * 1 - selected
         * 2 - not selected, mouse over
         * 3 - selected, mouse over
         * (+2 for mouse over)
         */
        public String name="";
        public int id=0;
        public int messagesNo=0;
        public int[] mType=new int[50];
        public String[] clanTag=new String[50];
        public int[] Security=new int[50];
        public String[] Username=new String[50];
        public String[] Message=new String[50];
        /*
         * 0 - normal
         * 1 - System Message
         * 2 - Client Only Message (Local System Message)
         */
        public RULabel object=new RULabel();
        
        public void tabMouseOver() { if(status+2 < 4) status+=2; }
        public void tabMouseOut() { if(status-2 >= 0) status-=2; }
        public void tabClicked(MouseEvent evt)
        {
        	int x=evt.getX();
        	int y=evt.getY();
        	Rectangle rect=new Rectangle();
        	rect.setBounds(24, 1, 6, 6);//the red x
        	if(rect.contains(x, y)) requestCloseChat(name);//exit the chatbox
        	else changeTab(id);//call change tab
        }
    }//end of tab class
    
    private Sockets socket;
    private MainPanel panel;
    public tab[] tabs=new tab[5];
    private int tabAmount=0;
    private boolean visible=true;
    private int tabSelected=6;
    public RULabel background=new RULabel();
    public RULabel joinTab=new RULabel();
    public BufferedImage[] tabImages=new BufferedImage[4];
    public RUScrollBar[] scrollbar={ new RUScrollBar(this), new RUScrollBar(this), new RUScrollBar(this), new RUScrollBar(this), new RUScrollBar(this) };
    private int[] scrollIndex=new int[5];
    private boolean isOpened=true;
    public RUButton arrButtonOpen=new RUButton();
    public RULabel arrBackgroundOpen=new RULabel();
    public RUButton arrButtonClosed=new RUButton();
    public RULabel arrBackgroundClosed=new RULabel();
    
    public void setVisible(boolean newVisible) { visible=newVisible; }
    public boolean getVisible() { return visible; }
    public void setTabSelected(int newTabSelected) { tabSelected=newTabSelected; }
    public int getTabSelected() { return tabSelected; }
    public int getTabAmount() { return tabAmount; }
    
    public RUChatbox(Sockets sock, MainPanel newPanel)
    {
    	socket=sock;
        panel=newPanel;
    	background.setWidth(430);
        background.setHeight(180);
        background.setX(370);
        background.setY(398);
        background.setCursorType(new Cursor(Cursor.TEXT_CURSOR));
        background.addMouseWheelListener(this);
        joinTab.setWidth(43);
        joinTab.setHeight(12);
        joinTab.setX(388);
        joinTab.setY(382);
        joinTab.setCursorType(new Cursor(Cursor.HAND_CURSOR));
        joinTab.addMouseListener(new MouseAdapter() { public void mouseClicked(MouseEvent evt) { joinTabClicked(); } });
        arrButtonOpen.setX(378);
        arrButtonOpen.setY(389);
        arrButtonOpen.addMouseListener(new MouseAdapter() { public void mouseClicked(MouseEvent evt) { setOpen(false); } });
		arrBackgroundOpen.setBackgroundColor(new Color(105, 105, 105));
		arrBackgroundOpen.setWidth(10);
		arrBackgroundOpen.setHeight(10);
		arrBackgroundOpen.setX(378);
		arrBackgroundOpen.setY(388);
		arrButtonClosed.setX(370);
		arrButtonClosed.setY(550);
		arrButtonClosed.addMouseListener(new MouseAdapter() { public void mouseClicked(MouseEvent evt) { setOpen(true); } });
		arrButtonClosed.setEnabled(false);
		arrBackgroundClosed.setBackgroundColor(new Color(105, 105, 105));
		arrBackgroundClosed.setWidth(30);
		arrBackgroundClosed.setHeight(30);
		arrBackgroundClosed.setX(370);
		arrBackgroundClosed.setY(550);
		arrBackgroundClosed.setEnabled(false);
        for(int i=0;i<5;i++)
        {
        	scrollbar[i].setVisible(false);
        	scrollbar[i].setWidth(20);
            scrollbar[i].setHeight(177);
            scrollbar[i].setX(368+(430-20));
            scrollbar[i].setY(400);
            scrollbar[i].setOptionsShown(13);
            scrollbar[i].setMaxValue(0);
        }
    }
    
    public int getChatID(String cName)
    {
        int i;
        for(i=0;i<tabAmount;i++)
        {
            if(cName.equals(tabs[i].name)) return i;
        }
        return 6;
    }
    
    public void changeTab(int newTab)
    {
    	if(tabs[tabSelected-1].status > 0) tabs[tabSelected-1].status--;
    	tabSelected=newTab+1;
		tabs[tabSelected-1].status++;
    }
    
    public void joinTabClicked()
    {
    	//ask for what chat, send it
    	String cName=RUMsgbox.inputBox(panel, "Enter the name of a chat you would like to join.", "Join A Chat");
    	if(cName == null) return;
    	if(cName == "") return;
    	requestOpenChat(cName);
    }
    
    public void addMessage(String list)//chatname|message type|security|username|message
    {
        //break the code up then call the other addMessage fnc
        int breakPoint;
        breakPoint=list.indexOf("|");
        if(breakPoint == -1) return;
        String cName=list.substring(0, breakPoint);
        list=list.substring(breakPoint+1);
        breakPoint=list.indexOf("|");
        if(breakPoint == -1) return;
        int cType=Integer.parseInt(list.substring(0, breakPoint));
        list=list.substring(breakPoint+1);
        breakPoint=list.indexOf("|");
        if(breakPoint == -1) return;
        String cTag=list.substring(0, breakPoint);
        list=list.substring(breakPoint+1);
        breakPoint=list.indexOf("|");
        if(breakPoint == -1) return;
        int cSecurity=Integer.parseInt(list.substring(0, breakPoint));
        list=list.substring(breakPoint+1); 
        breakPoint=list.indexOf("|");
        if(breakPoint == -1) return;
        String cUsername=list.substring(0, breakPoint);
        list=list.substring(breakPoint+1);
        String cMessage=list.substring(0);
        if(cType == 0) addMessage(cName, cType, cTag, cSecurity, cUsername, cMessage);
        else addSystemMessage(cName, cType, cMessage);
        int sChatID=getChatID(cName);
        scrollbar[sChatID].setMaxValue(tabs[sChatID].messagesNo);
        if(tabs[sChatID].messagesNo == 13) scrollbar[sChatID].setVisible(true);
        else if(tabs[sChatID].messagesNo < 13) scrollbar[sChatID].setVisible(false);
    }
    
    public void addSystemMessage(String cName, int cType, String cMessage)
    {
        //add a system message to a chatbox
        addMessage(cName, cType, "", 0, "", cMessage);
    }
    
    public void addMessage(String cName, int cType, String cTag, int cSecurity, String cUsername, String cMessage)
    {
        //add a message to a chatbox
        int chatID=getChatID(cName);
        int i;
        int tempMType=cType;
        String tempTag=cTag;
        int tempSecurity=cSecurity;
        String tempUsername=cUsername;
        String tempMessage=cMessage;
        /*shift the data:
        buffer the message for the buffer, set the new message, set the buffer for next round
        */
        for(i=49;i>=(tabs[chatID].messagesNo>=50? 0:49-tabs[chatID].messagesNo);i--)//dont have to worry about the last message
        {
            //buffers
            int oldMType=tabs[chatID].mType[i];
            String oldTag=tabs[chatID].clanTag[i];
            int oldSecurity=tabs[chatID].Security[i];
            String oldUsername=tabs[chatID].Username[i];
            String oldMessage=tabs[chatID].Message[i];
            //set the new messages
            tabs[chatID].mType[i]=tempMType;
            tabs[chatID].clanTag[i]=tempTag;
            tabs[chatID].Security[i]=tempSecurity;
            tabs[chatID].Username[i]=tempUsername;
            tabs[chatID].Message[i]=tempMessage;
            //set the other buffers
            tempMType=oldMType;
            tempTag=oldTag;
            tempSecurity=oldSecurity;
            tempUsername=oldUsername;
            tempMessage=oldMessage;
        }
        //increment the message counter
        tabs[chatID].messagesNo++;
    }
    
    public boolean addChat(String newName)
    {
    	if(tabAmount >= 5)
        {
            RUMsgbox.msgBox(null, "You have reached the maximum number of chats you are allowed to have.", "Chat Limit Reached");
            return false;
        }
        else
        {
            tabs[tabAmount]=new tab();
            tabs[tabAmount].name=newName;
            for(int i=0;i<tabAmount;i++) tabs[i].status=0;
            tabs[tabAmount].status=1;
            tabs[tabAmount].id=tabAmount;
            tabs[tabAmount].messagesNo=0;
            tabs[tabAmount].object.setSize(36, 17);
            tabs[tabAmount].object.setLocation(800-(8+((tabAmount+1)*37)), 382);//37 on purpose
            tabs[tabAmount].object.setCursorType(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            scrollbar[tabAmount].setMaxValue(0);
            tabAmount++;
            addSystemMessage(newName, 2, "You have joined "+newName+"!");
            tabSelected=tabAmount;
            return true;
        }
    }
    
    public void requestOpenChat(String cName)//keep it neat and ask the server, then it will be handled in the MainPanel
    {
    	String writeSocketData="O|"+cName;
        socket.write(writeSocketData);
    }
    
    public void requestCloseChat(String cName)//keep it neat and ask the server, then it will be handled in the MainPanel
    {
    	String writeSocketData="E|"+cName;
        socket.write(writeSocketData);
    }
    
    public int closeChat(String cName)
    {
        int index=6;
        if(tabAmount > 0)
        {
            index=getChatID(cName);
            tabs[index]=new tab();
            if(index < 6)
            {
                int i;
            	if(index < tabAmount-1)//no shifting needed
                {
                    for(i=index;i<tabAmount-1;i++)//shift them
                    {
                    	tabs[i]=tabs[i+1];
                    	tabs[i].id--;
                    	tabs[i].object.setLocation(800-(8+((i+1)*37)), 382);
                    }
                }
                tabAmount--;
                if(tabSelected >= tabAmount)
                {
                	tabSelected=1;
                	tabs[tabSelected-1].status=1;
                }
            }
        }
        return index;
    }
    
    public void drawChat(Graphics2D g)
    {
        if(visible == true)
        {
            if(isOpened)
            {
            	//draw the chat
                background.drawComponent(g);//the background
                joinTab.drawComponent(g);//the join tab
                arrBackgroundOpen.drawComponent(g);
                arrButtonOpen.drawComponent(g);
                g.setColor(new Color(128, 0, 0));
                Font font=new Font("Impact", 0, 10);
                g.setFont(font);
                g.drawString("Join Chat", 391, 396);
                //the rest of the tabs
                font=new Font("Lucida Console", 0, 10);
                for(int i=0;i<tabAmount;i++)
                {
                    g.drawImage(tabImages[tabs[i].status], tabs[i].object.getX(), tabs[i].object.getY(), tabs[i].object.getWidth(), tabs[i].object.getHeight(), null);
                    String drawText=tabs[i].name;
                    FontRenderContext frc=g.getFontRenderContext();
                    double mainStrWidth=font.getStringBounds(drawText, frc).getWidth();
                    double strWidth=0;
                    int endCounter=drawText.length();
                    //cut it down
                    do
                    {
                    	try { strWidth=font.getStringBounds(drawText.substring(0, endCounter), frc).getWidth(); }
                        catch(StringIndexOutOfBoundsException err) { System.out.println(tabs[i].object.getWidth()); break; }
                        if(strWidth >= tabs[i].object.getWidth()-4) endCounter--;
                    } while(strWidth >= tabs[i].object.getWidth()-4);
                    drawText=drawText.substring(0, endCounter);
                    //add room for ... if it was too long, width of 5
                    if(mainStrWidth > strWidth)
                    {
                        endCounter=drawText.length();
                        do
                        {
                            strWidth=font.getStringBounds(drawText.substring(0, endCounter)+"...", frc).getWidth();
                            if(strWidth >= tabs[i].object.getWidth()-4) endCounter--;
                        } while(strWidth >= tabs[i].object.getWidth()-4);
                        drawText=drawText.substring(0, endCounter);
                        //now that room was made add the ...
                        drawText+="...";
                    }
                    //shadow
                    //Color color=new Color(0, 0, 0);
                    //g.setColor(color);
                    //g.drawString(drawText, tabs[i].object.getX()+4+1, tabs[i].object.getY()+13+1);
                    //main text
                    Color color=new Color(128, 0, 0);
                    g.setColor(color);
                    g.drawString(drawText, tabs[i].object.getX()+4, tabs[i].object.getY()+13);
                }
                if(tabAmount > 0)
                {
                    //draw the messages
                    int i;
                    int extraLines=0;
                    try
                    {
                    	for(i=49;i>(tabs[tabSelected-1].messagesNo>=50? 0:49-tabs[tabSelected-1].messagesNo);i--)
                        {
                            String displayText;
                            int cType=tabs[tabSelected-1].mType[i-scrollIndex[tabSelected-1]];
                            String cTag=tabs[tabSelected-1].clanTag[i-scrollIndex[tabSelected-1]];
                            int cSecurity=tabs[tabSelected-1].Security[i-scrollIndex[tabSelected-1]];
                            String cUsername=tabs[tabSelected-1].Username[i-scrollIndex[tabSelected-1]];
                            String cMessage=tabs[tabSelected-1].Message[i-scrollIndex[tabSelected-1]];
                            if(cType == 0)//a person
                            {
                                String staffPrefix;
                                switch(cSecurity)
                                {
                                    case 0: staffPrefix=""; break;//member
                                    case 1: staffPrefix="[SFX]"; break;//sound effects arist
                                    case 2: staffPrefix="[J]"; break;//journalists
                                    case 3: staffPrefix="[GFX]"; break;//graphic artists
                                    case 4: staffPrefix="[MAP]"; break;//mappers
                                    case 5: staffPrefix="[MOD]"; break;//moderators
                                    case 6: staffPrefix="[SCRIPT]"; break;//scripters
                                    case 7: staffPrefix="[ADM]"; break;//administrators
                                    case 8: staffPrefix="[DEV]"; break;//developers
                                    default: staffPrefix=""; break;//no idea
                                }
                                String clanTag="";
                                if(cTag.length() > 0) clanTag="<"+cTag+">";
                                displayText=staffPrefix+cUsername+clanTag+": "+cMessage;
                            }
                            else if(cType == 1)
                            {
                                //system message
                                displayText="System: "+cMessage;
                            }
                            else//if(cType == 2)
                            {
                                //client only system message
                                displayText="Client: "+cMessage;
                            }
                            font=new Font("Lucida Console", 0, 12);
                            g.setFont(font);
                            Color color;
                            //display it
                            FontRenderContext frc=g.getFontRenderContext();
                            double mainStrWidth=font.getStringBounds(displayText, frc).getWidth();
                            double strWidth=mainStrWidth;
                            int counter=0;
                            int endCounter=0;
                            int msgLines=0;
                            do
                            {
                            	endCounter=displayText.length();
                                String line="";
                                do
                                {
                                	line=displayText.substring(counter, endCounter);
                                	strWidth=font.getStringBounds(line, frc).getWidth();
                                    if(strWidth >= 424-20) endCounter--;
                                } while(strWidth >= 424-20);
                                //shadow
                                color=new Color(0, 0, 0);
                                g.setColor(color);
                                g.drawString(line, 373+1, 411+(((49-i)+(extraLines))*13)+(msgLines*13)+1);
                                //main text
                                if(cType == 0) color=new Color(255, 255, 255);
                                else color=new Color(255, 165, 0);
                                g.setColor(color);
                                g.drawString(line, 373, 411+(((49-i)+(extraLines))*13)+(msgLines*13));
                                counter=endCounter;
                                msgLines++;
                                if(((msgLines-1)+extraLines+(49-i)) >= 12) break;//thats all that will fit
                            } while(endCounter < displayText.length());
                            extraLines+=msgLines-1;
                            if((extraLines+(49-i)) >= 12) break;//thats all that will fit
                            
                        }
                    } catch(ArrayIndexOutOfBoundsException e) { }
                    scrollbar[tabSelected-1].drawComponent(g);
                }
                else//no tabs open
                {
                	String displayText="Client: No chats open!";
                	font=new Font("Lucida Console", 0, 12);
                    g.setFont(font);
                    Color color;
                    //shadow
                    color=new Color(0, 0, 0);
                    g.setColor(color);
                    g.drawString(displayText, 373+1, 411+1);
                    //main text
                    color=new Color(255, 165, 0);
                    g.setColor(color);
                    g.drawString(displayText, 373, 411);
                }
            }
            else//closed
            {
            	arrBackgroundClosed.drawComponent(g);
                arrButtonClosed.drawComponent(g);
            }
        }
    }
    
    public void scrollUp(int index)
    {
    	if(scrollIndex[index] > 0) scrollIndex[index]--;
    	scrollbar[index].setValue(scrollIndex[index]);
    }
    
    public void mouseWheelMoved(MouseWheelEvent e)
    {
    	if(e.getUnitsToScroll() > 0) scrollDown(tabSelected-1);//down
    	else if(e.getUnitsToScroll() < 0) scrollUp(tabSelected-1);//up
    }
    
    public void scrollDown(int index)
    {
    	if(scrollIndex[index] < scrollbar[index].getMaxValue()-scrollbar[index].getOptionsShown()) scrollIndex[index]++;
    	scrollbar[index].setValue(scrollIndex[index]);
    }
    
    public void toggleChat() { setOpen(!isOpened); }
    
    public void setOpen(boolean newOpen)
    {
    	isOpened=newOpen;
		background.setEnabled(isOpened);
		joinTab.setEnabled(isOpened);
		arrButtonOpen.setEnabled(isOpened);
		arrBackgroundOpen.setEnabled(isOpened);
		arrButtonClosed.setEnabled(!isOpened);
		arrBackgroundClosed.setEnabled(!isOpened);
		for(int i=0;i<tabAmount;i++)
		{
			tabs[i].object.setEnabled(isOpened);
		}
    }
    public boolean getOpen() { return isOpened; }
}

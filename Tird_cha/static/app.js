class Chatbox{

    constructor(){

        this.args={
            openButton: document.querySelector('.chatbox__button'),         
            ChatBox: document.querySelector('.chatbox__support'),         
            sendButton: document.querySelector('.send__button')          
        }

        this.state=false;
        this.messages=[];

        }

    display(){
        const{openButton,ChatBox,sendButton}=this.args;
        openButton.addEventListener('click',()=>this.toggleState(ChatBox))
        sendButton.addEventListener('click',()=>this.openButton(ChatBox))

        const node =ChatBox.querySelector('input')
        node.addEventListener('keyup',({key})=>{
            if (key=='Enter'){
                this.openButton(ChatBox)
            }
            
            
        })
    }

    toggleState(Chatbox){
        this.state=!this.state;
        if (this.state){
            Chatbox.classList.add('chatbox--active')
        }   else{

             Chatbox.classList.remove('chatbox--active')
        }
           
    }

     oneSendButton(chatbox){
        var textField=chatbox.querySelector('input')
        let text1=textField.value
        if(text1==""){
            return;
        }


        let msg1={name:"User",message: text1}
        this.messages.push(msg1);

        // her the url local host
        fetch('the url local host'+'/predict',{
            method:"POST",
            body:JSON.stringify({message:text1}),
            mode:'core',
            Headers:{
                'Content-Type':'application/json'
            },
        })
        .then(r=> r.json())
        .then(r=> {
            let msg2={name:"THIRD",message:r.answer};
            this.messages.push(msg2)
            this.updateChatText(Chatbox)
            textField.value=''
        })
        .catch((error)=>{
            console.error("Error",error);
            this.updateChatText(chatbox)
            textField.value=''
        });
        }

        updateChatText(chatbox){
            var html='';
            this.messages.slice().reverse().forEach(function(item,){

                if (item.name=="THIRD")
                {
                    html+='<div class="messages__item messages__item--visitor">'+item.message+ '</div>'
                   
                }
                else{
                    html+='<div class="messages__item messages__item--operator>"'+item.message+'</div>'
                }

            });
            const chatmessage=chatbox.querySelector('.chatbox__messages');
            chatmessage.innerHTML=html;
        }

     }    

const chatbox=new chatbox();
chatbox.display()



    



